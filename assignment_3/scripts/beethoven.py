import ps_utils
import numpy as np
import matplotlib.pyplot as plt
import pickle

from enum import Enum


class Datasets(Enum):
    BEETHOVEN = 'Beethoven'
    MAT = 'mat_vase'
    SHINY = 'shiny_vase'
    SHINY2 = 'shiny_vase2'
    BUDDHA = 'Buddha'
    FACE = 'face'


def main(ransac=False, savefig=False, unbiased_integrate=True):
    dataset = Datasets.FACE.value
    path = f'src/{dataset}_{"ransac" if ransac else "woodham"}'

    I, mask, S = ps_utils.read_data_file(dataset)
    count_Is = I.shape[2]

    nz = np.nonzero(mask)
    count_nz = len(nz[0])

    J = np.vstack([I[:, :, i][nz] for i in range(count_Is)])
    assert (J.shape == (count_Is, count_nz))

    if ransac:
        ransac_threshold = 10.
        ransac_data = (J, S)
        M, inliers, fit = ps_utils.ransac_3dvector(ransac_data, ransac_threshold)
        pickle.dump((M, inliers, fit), open(f'src/ransac_runs/ransac_3dvector_{dataset}.sav', 'wb'))
    else:
        try:
            M = np.linalg.inv(S) @ J
        except np.linalg.LinAlgError:
            M = np.linalg.pinv(S) @ J

    Rho = np.linalg.norm(M, axis=0)
    N = M / np.tile(Rho, (M.shape[0], 1))

    albedo = np.zeros(mask.shape)
    albedo[nz] = Rho
    plt.title(f'Albedo {dataset}')
    plt.imshow(albedo)
    plt.axis('off')
    if savefig:
        plt.savefig(f'{path}_albedo.png')
    plt.show()

    n1 = np.zeros(mask.shape)
    n2 = np.zeros(mask.shape)
    n3 = np.zeros(mask.shape)

    n1[nz] = N[0, :]
    n2[nz] = N[1, :]
    n3[nz] = N[2, :]

    _, (ax1, ax2, ax3) = plt.subplots(1, M.shape[0])
    ax1.imshow(n1)
    ax1.set_title(r'N1')
    ax1.axis('off')
    ax2.imshow(n2)
    ax2.set_title(r'N2')
    ax2.axis('off')
    ax3.imshow(n3)
    ax3.set_title(r'N3')
    ax3.axis('off')
    plt.tight_layout()
    plt.suptitle(f'Normals {dataset}')
    if savefig:
        plt.savefig(f'{path}_normals.png')
    plt.show()

    if unbiased_integrate:
        z = ps_utils.unbiased_integrate(n1, n2, n3, mask)
    else:
        z = ps_utils.simchony_integrate(n1, n2, n3, mask)
    # z = ps_utils.simchony_integrate(n1, n2, n3, mask)
    ps_utils.display_surface(z)


if __name__ == '__main__':
    main(ransac=True,
         savefig=True,
         unbiased_integrate=True)
