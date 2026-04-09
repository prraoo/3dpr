import numpy as np

class LightSampler:
    def __init__(self, light_path, width, height):
        self.width = width
        self.height = height
        self.light_path = light_path
        self.light_vectors = np.load(light_path)

    @staticmethod
    def normalize(v):
        norm = np.linalg.norm(v)
        return v / norm if norm != 0 else v

    @staticmethod
    def cartesian_to_spherical(v):
        x, y, z = v
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arctan2(y, x)
        phi = np.arccos(z / r)
        return theta, phi

    def spherical_to_uv(self, theta, phi):
        u = (theta + np.pi) / (2 * np.pi) * self.width
        v = phi / np.pi * self.height
        return int(u) % self.width, int(v) % self.height

    def sample_pixels(self, emap):
        sampled_pixels = []
        for vec in self.light_vectors:
            norm_vec = self.normalize(vec)
            theta, phi = self.cartesian_to_spherical(norm_vec)
            u, v = self.spherical_to_uv(theta, phi)
            sampled_pixels.append(emap[v, u])
        return np.array(sampled_pixels)


if __name__ == '__main__':
    # Example usage
    light_path = '/HPS/prao2/static00/datasets/OLAT_c2-Multiple-IDs/raw/vorf_gan_config/light_dirs.npy'
    width, height = 20, 10

    light_sampler = LightSampler(light_path, width, height)
    emap = np.zeros((10,20,3))

    # Sample pixels
    sampled_pixels = light_sampler.sample_pixels(emap)

    # Output sampled pixel coordinates
    sampled_pixels = sampled_pixels.reshape(-1, 3)
    print(sampled_pixels.shape)
    emap = np.tile(sampled_pixels, (512, 512, 1, 1)).transpose(2, 0, 1, 3)
    print(emap.shape)
