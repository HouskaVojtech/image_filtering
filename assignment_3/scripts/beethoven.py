import ps_utils
import numpy as np
from matplotlib import pyplot as plt


def main(file_name='Beethoven'):
    # read Beethoven data
    I, mask, S = ps_utils.read_data_file(file_name)

    # get indices of non zero pixels in mask
    nz = np.argwhere(mask > 0)
    m, n = mask.shape

    flat_mask = mask.flatten()
    Is = []
    for i in range(3):
        Is.append(I[:, :, i].flatten())

    stacked_I = np.vstack(Is)

    J = stacked_I[:, flat_mask > 0]

    M = np.dot(np.linalg.inv(S), J)

    # The part bellow is not mine

    # get albedo as norm of M and normalize M
    Rho = np.linalg.norm(M, axis=0)
    N = M / np.tile(Rho, (3, 1))

    n1 = np.zeros((m, n))
    n2 = np.zeros((m, n))
    n3 = np.ones((m, n))

    n1[nz] = N[0, :]
    n2[nz] = N[1, :]
    n3[nz] = N[2, :]

    _, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(n1)
    ax1.axis('off')
    ax2.imshow(n2)
    ax2.axis('off')
    ax3.imshow(n3)
    ax3.axis('off')
    plt.tight_layout()
    plt.show()

    z = ps_utils.unbiased_integrate(n1, n2, n3, mask)
    ps_utils.display_surface_matplotlib(z)


if __name__ == '__main__':
    main()
    # main('mat_vase')
