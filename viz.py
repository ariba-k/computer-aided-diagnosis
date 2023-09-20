import os
import nibabel as nib
from nilearn import plotting
from data import get_data

# Paths and Data Loading
base_path = "data/alzheimer/OASIS"
base_image_path = os.path.join(base_path, "OAS2_RAW_PART1")
base_csv_path = os.path.join(base_path, "oasis_longitudinal_demographics.xlsx")

paths, labels, df = get_data(base_csv_path)

# Label encoding map
label_encoding = {'Converted': 0, 'Demented': 1, 'Nondemented': 2}


def plot_3d_image(file_path, title):
    """
    Plot and save a 3D image given a file path and title.
    """
    img = nib.load(file_path)
    display = plotting.plot_anat(img, title=title)
    display.savefig(f"viz/{title}.png")  # Save the figure
    plotting.show()  # Display the figure in a new window


def main():
    # Get the first occurrence index for each distinct label
    distinct_label_indices = [labels[labels == label].index[0] for label in label_encoding.values()]

    # Retrieve the corresponding paths for these indices
    distinct_label_paths = [paths[i] for i in distinct_label_indices]

    # Plot 3D images for the selected paths
    for label, path in zip(label_encoding.keys(), distinct_label_paths):
        plot_3d_image(path, label)


if __name__ == "__main__":
    main()
