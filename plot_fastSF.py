import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_structure_functions(file_path="SF_Grid_pll.h5", orders=[3, 5, 7], domain_size=1.0):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found. Run fastSF first!")
        return

    plt.figure(figsize=(10, 7))
    
    with h5py.File(file_path, "r") as f:
        # We'll use this to scale our reference line to match your data's magnitude
        ref_l = []
        ref_s3 = []

        for q in orders:
            dataset_name = f"SF_Grid_pll{q}"
            if dataset_name not in f:
                print(f"Warning: Order q={q} not found in {file_path}. Skipping.")
                continue
            
            data = f[dataset_name][:]
            nx, ny, nz = data.shape
            
            dx, dy, dz = domain_size/nx, domain_size/ny, domain_size/nz
            x = np.arange(nx) * dx
            y = np.arange(ny) * dy
            z = np.arange(nz) * dz
            
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            distances = np.sqrt(X**2 + Y**2 + Z**2)
            
            flat_distances = distances.ravel()
            flat_data = data.ravel()
            
            valid_mask = (flat_distances > 0) & (flat_data > 0)
            flat_distances = flat_distances[valid_mask]
            flat_data = flat_data[valid_mask]
            
            num_bins = int(np.max([nx, ny, nz]) * 0.75)
            bins = np.linspace(flat_distances.min(), flat_distances.max(), num_bins)
            bin_centers = 0.5 * (bins[1:] + bins[:-1])
            
            counts, _ = np.histogram(flat_distances, bins=bins)
            sums, _ = np.histogram(flat_distances, bins=bins, weights=flat_data)
            
            mean_sf = np.divide(sums, counts, out=np.zeros_like(sums), where=counts!=0)
            valid_bins = (counts > 0) & (mean_sf > 0)
            
            line, = plt.loglog(bin_centers[valid_bins], mean_sf[valid_bins], label=f'q = {q}', lw=2)
            
            # Capture q=3 data to align the reference line
            if q == 3:
                ref_l = bin_centers[valid_bins]
                ref_s3 = mean_sf[valid_bins]

    # Add Kolmogorov 4/5 Law Reference Line (Slope = 1)
    if len(ref_l) > 0:
        # We pick a point in the middle of your data to anchor the reference line
        mid = len(ref_l) // 2
        anchor_l = ref_l[mid]
        anchor_s3 = ref_s3[mid]
        
        # S3 ~ l^1 (Kolmogorov)
        # We draw it across the same range as your data
        kolmo_line = anchor_s3 * (ref_l / anchor_l)**1.0
        plt.loglog(ref_l, kolmo_line, 'k--', alpha=0.7, label='Kolmogorov Slope (l^1)')

    plt.xlabel(r'Separation scale $l$ (Log Scale)', fontsize=14)
    plt.ylabel(r'$S_q(l)$ (Log Scale)', fontsize=14)
    plt.title('Longitudinal Velocity Structure Functions (fastSF)', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.3)
    
    output_name = "SF_comparison.png"
    plt.savefig(output_name, dpi=300)
    print(f"Success! Plot saved as {output_name}")
    plt.show()

if __name__ == "__main__":
    plot_structure_functions(orders=[3, 5, 7], domain_size=1.0)
