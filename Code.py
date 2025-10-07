'''
Team Members:
    CS24MTECH14013 : Mohit Manoj Patil
    CS24MTECH14010: Veeresh Shukla
    CS24MTECH12018 : Deeba Afridi
'''

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns
from scipy.spatial.distance import cdist
import os

# Ensure TensorFlow doesn't allocate all GPU memory if not needed
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)

# --- VAE Model Definition ---
# (Keeping the core VAE structure as requested, but distinct naming)
class VariationalAutoencoder(keras.Model):
    """
    Defines the Variational Autoencoder model structure.
    Follows the standard VAE architecture with encoder, decoder, and sampling.
    """
    def __init__(self, original_dim, latent_dim, intermediate_dims=[128, 64], name="vae", **kwargs):
        super(VariationalAutoencoder, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.latent_dim = latent_dim
        self.intermediate_dims = intermediate_dims

        # Build Encoder Components
        encoder_inputs = keras.Input(shape=(original_dim,))
        x = encoder_inputs
        for dim in self.intermediate_dims:
            x = layers.Dense(dim, activation="relu")(x)
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_variance = layers.Dense(latent_dim, name="z_log_variance")(x)

        # Sampling layer
        self.z = self._create_sampling_layer()([z_mean, z_log_variance])
        self.encoder = keras.Model(encoder_inputs, [z_mean, z_log_variance, self.z], name="encoder")

        # Build Decoder Components
        latent_inputs = keras.Input(shape=(latent_dim,))
        x = latent_inputs
        for dim in reversed(self.intermediate_dims):
            x = layers.Dense(dim, activation="relu")(x)
        decoder_outputs = layers.Dense(original_dim, activation="linear")(x) # Linear activation for reconstruction
        self.decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    def _create_sampling_layer(self):
        """Creates the Lambda layer for sampling from the latent distribution."""
        def sampling(args):
            z_mean, z_log_variance = args
            batch_size = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch_size, dim))
            return z_mean + tf.exp(0.5 * z_log_variance) * epsilon
        return layers.Lambda(sampling, name='sampling_lambda')

    def call(self, inputs):
        z_mean, z_log_variance, z_sampled = self.encoder(inputs)
        reconstructed_outputs = self.decoder(z_sampled)
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_variance - tf.square(z_mean) - tf.exp(z_log_variance) + 1
        )
        self.add_loss(kl_loss)
        return reconstructed_outputs

    def encode_data(self, data):
        _, _, z_latent = self.encoder.predict(data)
        return z_latent

    def decode_data(self, latent_representation):
        return self.decoder.predict(latent_representation)


# --- Main Outlier Detection Class ---
class VAEKMeansOutlierDetector:
    """
    Encapsulates the process of detecting outliers using VAE and K-Means.

    Steps:
    1. Load and preprocess data.
    2. Train VAE models with different latent dimensions.
    3. Apply K-Means clustering on latent representations.
    4. Use the Elbow method to help determine the number of clusters.
    5. Select the best VAE latent dimension and K-Means cluster count based on silhouette score.
    6. Identify outliers as points far from their cluster centroids in the chosen latent space.
    7. Visualize results and save outliers.
    """
    def __init__(self, file_path, numeric_feature_cols=None,
                 latent_dim_options=[2, 3, 5, 8], max_clusters_to_try=10,
                 vae_epochs=50, vae_batch_size=32,
                 outlier_percentile=95, output_dir="results"):
        """
        Initializes the detector configuration.

        Args:
            file_path (str): Path to the input CSV data file.
            numeric_feature_cols (list, optional): List of column names to use.
                                                  If None, selects all numeric columns. Defaults to None.
            latent_dim_options (list): List of latent dimensions to try for the VAE.
            max_clusters_to_try (int): Max number of clusters for the elbow method.
            vae_epochs (int): Number of epochs for VAE training.
            vae_batch_size (int): Batch size for VAE training.
            outlier_percentile (int): Percentile threshold for distance to identify outliers.
            output_dir (str): Directory to save results (plots, outlier file).
        """
        self.file_path = file_path
        self.numeric_feature_cols = numeric_feature_cols
        self.latent_dim_options = latent_dim_options
        self.max_clusters_to_try = max_clusters_to_try
        self.vae_epochs = vae_epochs
        self.vae_batch_size = vae_batch_size
        self.outlier_percentile = outlier_percentile
        self.output_dir = output_dir

        # Internal state variables
        self.original_dataframe = None
        self.processed_data = None
        self.feature_scaler = None
        self.best_latent_dim = None
        self.best_cluster_count = None
        self.best_silhouette_score = -1  # Silhouette score is between -1 and 1
        self.best_vae_model = None
        self.best_kmeans_model = None
        self.best_latent_representation = None
        self.training_results = [] # To store results for each config
        self.outlier_indices = None
        self.outlier_details_df = None

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Results will be saved in: {self.output_dir}")


    def _load_and_prepare_data(self):
        """Loads data, selects features, and applies scaling."""
        print(f"Loading data from {self.file_path}...")
        try:
            df = pd.read_csv(self.file_path)
            self.original_dataframe = df.copy() # Keep original for reference

            if self.numeric_feature_cols:
                # Check if provided columns exist and are numeric
                missing_cols = [col for col in self.numeric_feature_cols if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"Specified columns not found in data: {missing_cols}")
                data_subset = df[self.numeric_feature_cols]
                non_numeric = data_subset.select_dtypes(exclude=['float64', 'int64']).columns
                if not non_numeric.empty:
                     raise ValueError(f"Specified columns contain non-numeric data: {list(non_numeric)}")
                numeric_data = data_subset.values
            else:
                # Auto-select numeric columns
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                if numeric_cols.empty:
                    raise ValueError("No numeric columns found in the data.")
                print(f"Using automatically detected numeric columns: {list(numeric_cols)}")
                numeric_data = df[numeric_cols].values
                # Store the names for potential later use if needed
                self.numeric_feature_cols = list(numeric_cols)

            # Handle potential NaN values (simple mean imputation)
            if np.isnan(numeric_data).any():
                print("Warning: NaN values detected. Imputing with column means.")
                col_means = np.nanmean(numeric_data, axis=0)
                nan_indices = np.where(np.isnan(numeric_data))
                numeric_data[nan_indices] = np.take(col_means, nan_indices[1])

            print(f"Data shape for processing: {numeric_data.shape}")

            # Normalize data
            self.feature_scaler = StandardScaler()
            self.processed_data = self.feature_scaler.fit_transform(numeric_data)
            print("Data loaded and preprocessed successfully.")

        except FileNotFoundError:
            print(f"Error: File not found at {self.file_path}")
            raise
        except ValueError as ve:
            print(f"Error during data preparation: {ve}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred during data loading: {e}")
            raise


    def _find_optimal_clusters_elbow(self, data_to_cluster):
        """
        Uses the elbow method to suggest an optimal number of clusters for K-Means.

        Args:
            data_to_cluster (np.ndarray): The data (e.g., latent representation) to cluster.

        Returns:
            int: Suggested optimal number of clusters.
        """
        distortions = []
        cluster_range = range(1, self.max_clusters_to_try + 1)

        print("Calculating distortions for Elbow Method:")
        for k in cluster_range:
            kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=10) # n_init suppresses warning
            kmeans_model.fit(data_to_cluster)
            # Calculate average sum of squared distances to nearest centroid (inertia is sum of squared)
            # distortion = kmeans_model.inertia_ / data_to_cluster.shape[0] # Average inertia
            # Using average Euclidean distance instead (like original code)
            distortion = sum(np.min(cdist(data_to_cluster, kmeans_model.cluster_centers_, 'euclidean'), axis=1)) / data_to_cluster.shape[0]
            distortions.append(distortion)
            print(f"  k={k}: Distortion = {distortion:.4f}")

        # Simple heuristic for finding the elbow: point with the largest decrease rate drop
        # Calculate percentage decrease
        if len(distortions) > 1:
            decreases = np.diff(distortions)
            relative_decreases = -decreases / distortions[:-1] # Percentage drop

            # Find the point where the rate of decrease significantly slows down
            # Look for the first point where the decrease is less than a fraction (e.g., 10%) of the previous max decrease
            if len(relative_decreases) > 1:
                 # Find the largest decrease first
                 # Then find where the next decrease is much smaller
                 # This is a simple heuristic and might need tuning
                 elbow_point = 2 # Default start
                 max_decrease = 0
                 for i in range(len(relative_decreases)):
                     if relative_decreases[i] > max_decrease:
                         max_decrease = relative_decreases[i]

                 # Find where decrease is less than 15% of max decrease
                 for i in range(len(relative_decreases)):
                     if relative_decreases[i] < max_decrease * 0.15:
                          elbow_point = i + 2 # +1 for index, +1 because diff reduces length
                          break
                 else: # If no significant drop, maybe take point before last major drop
                      elbow_point = np.argmax(relative_decreases) + 2 if relative_decreases.any() else 3


            else: # Only two points
                 elbow_point = 2

            # Ensure elbow point is reasonable
            elbow_point = max(2, min(elbow_point, self.max_clusters_to_try))

        else: # Only one cluster tried
            elbow_point = 2 # Default if only k=1 was tried

        print(f"Suggested optimal number of clusters (Elbow Method): {elbow_point}")
        return elbow_point


    def _train_evaluate_and_select_best_model(self):
        """Trains VAEs for different latent dimensions and evaluates clustering."""
        if self.processed_data is None:
            raise RuntimeError("Data not loaded or prepared. Call _load_and_prepare_data first.")

        input_dimension = self.processed_data.shape[1]
        self.best_silhouette_score = -1 # Reset best score

        print("\n--- Starting VAE Training and Clustering Evaluation ---")
        for l_dim in self.latent_dim_options:
            print(f"\nEvaluating Latent Dimension: {l_dim}")

            # 1. Build and Train VAE
            print("  Building and training VAE...")
            vae = VariationalAutoencoder(original_dim=input_dimension, latent_dim=l_dim)
            vae.compile(optimizer=keras.optimizers.Adam())
            # Consider adding validation split for monitoring, though not strictly needed for unsupervised
            vae.fit(self.processed_data, self.processed_data,
                    epochs=self.vae_epochs,
                    batch_size=self.vae_batch_size,
                    shuffle=True,
                    verbose=0) # Keep training quiet
            print("  VAE training complete.")

            # 2. Encode data to latent space
            current_latent_representation = vae.encode_data(self.processed_data)
            print(f"  Encoded data to {current_latent_representation.shape} latent space.")

            # 3. Determine optimal cluster count using Elbow Method
            suggested_k = self._find_optimal_clusters_elbow(current_latent_representation)

            # 4. Evaluate K-Means around the suggested K using Silhouette Score
            # Define a range around the suggested K
            k_range_to_test = range(max(2, suggested_k - 1), suggested_k + 3)
            print(f"  Evaluating K-Means for k in {list(k_range_to_test)}...")

            for k_clusters in k_range_to_test:
                if k_clusters > current_latent_representation.shape[0] -1: # Need at least k+1 samples for silhouette
                    print(f"    Skipping k={k_clusters} (not enough samples for silhouette score).")
                    continue
                if k_clusters < 2: # Silhouette requires at least 2 clusters
                     print(f"    Skipping k={k_clusters} (silhouette requires >= 2 clusters).")
                     continue

                # Apply K-Means
                kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
                cluster_assignments = kmeans.fit_predict(current_latent_representation)

                # Calculate Silhouette Score
                try:
                    silhouette_avg = silhouette_score(current_latent_representation, cluster_assignments)
                    print(f"    k={k_clusters}: Silhouette Score = {silhouette_avg:.4f}")

                    result_entry = {
                        'latent_dim': l_dim,
                        'n_clusters': k_clusters,
                        'silhouette_score': silhouette_avg
                    }
                    self.training_results.append(result_entry)

                    # Check if this configuration is the best so far
                    if silhouette_avg > self.best_silhouette_score:
                        self.best_silhouette_score = silhouette_avg
                        self.best_latent_dim = l_dim
                        self.best_cluster_count = k_clusters
                        self.best_vae_model = vae
                        self.best_kmeans_model = kmeans
                        self.best_latent_representation = current_latent_representation
                        print(f"    *** New best configuration found! ***")

                except ValueError as e:
                     print(f"    k={k_clusters}: Could not calculate silhouette score ({e}). Skipping.")


        if self.best_vae_model is None:
            raise RuntimeError("Failed to find a suitable VAE/K-Means configuration. Check data and parameters.")

        print("\n--- Model Selection Complete ---")
        print(f"Best Latent Dimension: {self.best_latent_dim}")
        print(f"Best Number of Clusters: {self.best_cluster_count}")
        print(f"Best Silhouette Score: {self.best_silhouette_score:.4f}")


    def _identify_outliers(self):
        """Identifies outliers based on distance to cluster centroids in the best latent space."""
        if self.best_latent_representation is None or self.best_kmeans_model is None:
            raise RuntimeError("Best model not determined. Run training and evaluation first.")

        print("\n--- Identifying Outliers ---")
        cluster_assignments = self.best_kmeans_model.labels_
        centroids = self.best_kmeans_model.cluster_centers_

        # Calculate Euclidean distance for each point to its assigned centroid
        distances_to_centroid = np.zeros(self.best_latent_representation.shape[0])
        for i in range(len(distances_to_centroid)):
            assigned_cluster_index = cluster_assignments[i]
            centroid_coords = centroids[assigned_cluster_index]
            point_coords = self.best_latent_representation[i]
            distances_to_centroid[i] = np.linalg.norm(point_coords - centroid_coords)

        # Determine the distance threshold using the specified percentile
        distance_threshold = np.percentile(distances_to_centroid, self.outlier_percentile)
        print(f"Using {self.outlier_percentile}th percentile distance threshold: {distance_threshold:.4f}")

        # Identify points exceeding the threshold
        self.outlier_indices = np.where(distances_to_centroid > distance_threshold)[0]
        num_outliers = len(self.outlier_indices)
        total_points = len(self.original_dataframe)
        outlier_percentage = (num_outliers / total_points * 100) if total_points > 0 else 0

        print(f"Identified {num_outliers} outliers ({outlier_percentage:.2f}% of total data).")

        if num_outliers > 0:
            # Create a DataFrame with details of the outliers
            self.outlier_details_df = self.original_dataframe.iloc[self.outlier_indices].copy()
            # Add outlier specific info
            self.outlier_details_df['latent_cluster'] = cluster_assignments[self.outlier_indices]
            self.outlier_details_df['distance_to_centroid'] = distances_to_centroid[self.outlier_indices]
            # Add latent space coordinates if needed (can make df very wide)
            # for ld in range(self.best_latent_dim):
            #    self.outlier_details_df[f'latent_dim_{ld+1}'] = self.best_latent_representation[self.outlier_indices, ld]

            print("\nSample Outlier Details:")
            print(self.outlier_details_df.head())
        else:
            self.outlier_details_df = pd.DataFrame() # Empty dataframe
            print("No outliers found based on the current threshold.")


    def visualize_latent_space(self, plot_filename="clusters_and_outliers_2d.png"):
        """
        Visualizes the clustering results in the best latent space (2D only).
        Outliers are shown with their cluster color fill and a red circle outline.
        """
        if self.best_latent_representation is None or self.best_kmeans_model is None:
            print("Warning: Cannot visualize - best model not available.")
            return

        if self.best_latent_dim != 2:
            print(f"Skipping visualization: Latent dimension is {self.best_latent_dim} (only 2D is supported for this plot).")
            return

        print("\n--- Generating Latent Space Visualization (2D) ---")
        plt.style.use('seaborn-v0_8-whitegrid') # Use a style with grid similar to example
        plt.figure(figsize=(12, 10))

        cluster_assignments = self.best_kmeans_model.labels_
        centroids = self.best_kmeans_model.cluster_centers_
        unique_labels = np.unique(cluster_assignments)

        # Use a palette similar to the example image ('tab10' is a good candidate)
        # Ensure enough colors if there are many clusters, fallback or cycle if needed
        if len(unique_labels) <= 10:
            palette = sns.color_palette("tab10", n_colors=len(unique_labels))
        else:
            # Fallback if more than 10 clusters needed
            palette = sns.color_palette("viridis", n_colors=len(unique_labels))

        # Define marker sizes and outline properties
        base_point_size = 50       # Size for regular points
        outlier_point_size = 55    # Size for the filled part of outlier points (can be same as base)
        outlier_ring_size = 120    # Size for the red ring outline (needs to be larger)
        outlier_linewidth = 1.5    # Thickness of the red ring
        point_alpha = 0.6          # Transparency for points

        # Plot points cluster by cluster
        legend_handles = [] # To manually create legend later if needed

        for i, label in enumerate(unique_labels):
            mask = (cluster_assignments == label)
            points_in_cluster = self.best_latent_representation[mask]

            # Separate outliers within this cluster
            is_outlier_mask = np.zeros(len(points_in_cluster), dtype=bool)
            if self.outlier_indices is not None:
                 # Find which points in *this specific cluster* are outliers
                 original_indices_in_cluster = np.where(mask)[0]
                 outliers_in_this_cluster_mask = np.isin(original_indices_in_cluster, self.outlier_indices)
                 is_outlier_mask = outliers_in_this_cluster_mask # Indices relative to points_in_cluster

            # Plot non-outliers for this cluster
            non_outlier_points = points_in_cluster[~is_outlier_mask]
            if len(non_outlier_points) > 0:
                # Add cluster label only to the first scatter plot for this color
                handle = plt.scatter(non_outlier_points[:, 0], non_outlier_points[:, 1],
                            color=palette[i],
                            label=f'Cluster {label}',
                            alpha=point_alpha,
                            s=base_point_size,
                            linewidths=0) # No edge for regular points
                #legend_handles.append(handle) # Add handle for legend if needed

            # Plot outliers for this cluster (filled point)
            outlier_points = points_in_cluster[is_outlier_mask]
            if len(outlier_points) > 0:
                plt.scatter(outlier_points[:, 0], outlier_points[:, 1],
                            color=palette[i], # Use the cluster's color
                            alpha=point_alpha,
                            s=outlier_point_size, # Can be same as base or slightly different
                            linewidths=0)

        # After plotting all filled points, plot the red rings ON TOP of the outliers
        if self.outlier_indices is not None and len(self.outlier_indices) > 0:
             outlier_handle = plt.scatter(self.best_latent_representation[self.outlier_indices, 0],
                         self.best_latent_representation[self.outlier_indices, 1],
                         s=outlier_ring_size,       # Larger size for the ring
                         facecolors='none',         # No fill for the ring itself
                         edgecolors='red',
                         linewidth=outlier_linewidth,
                         label='Outliers')          # Label for the red ring in the legend
             #legend_handles.append(outlier_handle)


        # Plot centroids
        centroid_handle = plt.scatter(centroids[:, 0], centroids[:, 1], s=200, marker='X', # Slightly smaller X?
                    c='black', label='Cluster Centers', zorder=10) # Ensure centroids are on top
        #legend_handles.append(centroid_handle)

        # Configure plot & legend
        plt.title(f'Clusters and Outliers in 2D Latent Space', fontsize=16) # Match example title
        plt.xlabel('Latent Dimension 1', fontsize=12)
        plt.ylabel('Latent Dimension 2', fontsize=12)
        plt.legend(loc='best', fontsize=10) # Automatic legend should work
        # plt.grid(True, linestyle='--', alpha=0.5) # Grid is handled by style 'seaborn-v0_8-whitegrid'
        plt.gca().set_aspect('equal', adjustable='box') # Make axes visually square

        save_path = os.path.join(self.output_dir, plot_filename)
        try:
            plt.savefig(save_path, dpi=150) # Increase dpi for better quality save
            print(f"Visualization saved to {save_path}")
        except Exception as e:
            print(f"Error saving visualization: {e}")
        plt.show()

    def save_outliers(self, filename="identified_outliers.csv"):
        """Saves the identified outliers (with original features and extra info) to a CSV file."""
        if self.outlier_details_df is None:
            print("No outlier details DataFrame available to save.")
            return
        if self.outlier_details_df.empty:
            print("No outliers were identified, skipping file save.")
            return

        save_path = os.path.join(self.output_dir, filename)
        try:
            self.outlier_details_df.to_csv(save_path, index=False)
            print(f"Outlier details successfully saved to {save_path}")
        except Exception as e:
            print(f"Error saving outlier data to CSV: {e}")


    def run_detection_pipeline(self):
        """Executes the full outlier detection pipeline."""
        try:
            self._load_and_prepare_data()
            self._train_evaluate_and_select_best_model()
            self._identify_outliers()
            self.visualize_latent_space() # Visualize the best result
            self.save_outliers()          # Save the identified outliers

            print("\n--- Outlier Detection Pipeline Finished ---")
            print(f"Total data points processed: {len(self.original_dataframe)}")
            if self.outlier_indices is not None:
                print(f"Number of outliers detected: {len(self.outlier_indices)}")
            print(f"Best model configuration: Latent Dim={self.best_latent_dim}, Clusters={self.best_cluster_count}, Silhouette={self.best_silhouette_score:.4f}")
            print(f"Results saved in '{self.output_dir}' directory.")

        except (FileNotFoundError, ValueError, RuntimeError, Exception) as e:
            print(f"\n--- Pipeline execution failed ---")
            print(f"Error: {e}")
            # Potentially add more specific error handling or logging

    # --- Getter Methods ---
    def get_original_data(self):
        return self.original_dataframe

    def get_processed_data(self):
        return self.processed_data

    def get_best_model_config(self):
        return {
            'latent_dim': self.best_latent_dim,
            'n_clusters': self.best_cluster_count,
            'silhouette_score': self.best_silhouette_score
        }

    def get_best_vae_model(self):
        return self.best_vae_model

    def get_best_kmeans_model(self):
        return self.best_kmeans_model

    def get_latent_representation(self):
        return self.best_latent_representation

    def get_outlier_details(self):
        return self.outlier_details_df

    def get_outlier_indices(self):
        return self.outlier_indices

# --- Main Execution ---
if __name__ == "__main__":
    # Configuration
    INPUT_FILE = "data.csv"
    LATENT_DIMS = [2, 4, 6, 8]
    OUTLIER_THRESHOLD_PERCENTILE = 95
    EPOCHS = 50
    BATCH_SIZE = 32
    OUTPUT_FOLDER = "vae_kmeans_outlier_results_v2" # Use a different output folder maybe

    # Create dummy data if 'data.csv' doesn't exist
    if not os.path.exists(INPUT_FILE):
        print(f"'{INPUT_FILE}' not found. Creating dummy data...")
        num_samples = 1000
        num_features = 10
        np.random.seed(42)
        # Create distinct clusters for better visualization
        centers = [[2, 2], [-2, -2], [2, -2]]
        cluster_std = 0.8
        cluster_data = []
        points_per_cluster = num_samples // 3
        for i in range(3):
             cluster_data.append(np.random.randn(points_per_cluster, num_features) * cluster_std + centers[i%len(centers)])
        normal_data = np.vstack(cluster_data)

        # Add some outliers far away
        num_outliers = max(10, int(num_samples * 0.05)) # Ensure some outliers
        outlier_data = np.random.uniform(-8, 8, size=(num_outliers, num_features))

        # Combine and shuffle
        data = np.vstack([normal_data[:num_samples-num_outliers], outlier_data]) # Make total num_samples
        np.random.shuffle(data)
        dummy_df = pd.DataFrame(data, columns=[f'feature_{i+1}' for i in range(num_features)])
        dummy_df.to_csv(INPUT_FILE, index=False)
        print(f"Dummy data created and saved to '{INPUT_FILE}'.")


    # Instantiate and run the detector
    detector = VAEKMeansOutlierDetector(
        file_path=INPUT_FILE,
        latent_dim_options=LATENT_DIMS,
        outlier_percentile=OUTLIER_THRESHOLD_PERCENTILE,
        vae_epochs=EPOCHS,
        vae_batch_size=BATCH_SIZE,
        output_dir=OUTPUT_FOLDER
    )

    detector.run_detection_pipeline()