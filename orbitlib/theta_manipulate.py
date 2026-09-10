import torch

def period_multiply(theta, T, periods: int, truncate=False, new_freq_cutoff=None):
    """Represent the same loop over multiple periods.

    ``new_freq_cutoff`` explicitly selects the output cutoff. Otherwise,
    ``truncate`` retains the original cutoff; the default expands it by
    ``periods`` so that all original Fourier modes are preserved.
    """

    frequency_cutoff = (theta.shape[1] - 1) // 2

    if new_freq_cutoff is None:
        new_freq_cutoff = (1 if truncate else periods) * frequency_cutoff

    new_shape = (theta.shape[0], 2 * new_freq_cutoff + 1)
    new_theta = torch.zeros(new_shape)

    new_theta[:, 0] = theta[:, 0]

    index_cutoff = min(new_freq_cutoff // periods, frequency_cutoff) + 1

    new_theta[:, periods : periods * index_cutoff : periods] = theta[:, 1 : index_cutoff]
    new_theta[
        :, new_freq_cutoff + periods : new_freq_cutoff + periods * index_cutoff : periods
    ] = theta[
        :, frequency_cutoff + 1 : frequency_cutoff + index_cutoff
    ]

    return new_theta, periods * T

def frequency_extend(theta, new_freq_cutoff, strict=True):
    """Extend a coefficient tensor to ``new_freq_cutoff`` with zero modes.

    When ``strict`` is true, reject a cutoff below the current cutoff.
    """
    frequency_cutoff = (theta.shape[1] - 1) // 2
    num_variables = theta.shape[0]

    if strict and new_freq_cutoff < frequency_cutoff:
        raise Exception('Extension cutoff below original cutoff')

    const = theta[:, 0:1]
    a_params = theta[:, 1:frequency_cutoff+1]
    b_params = theta[:, frequency_cutoff+1:]

    return torch.cat((
        const,
        a_params,
        torch.zeros((num_variables, new_freq_cutoff - frequency_cutoff)),
        b_params,
        torch.zeros((num_variables, new_freq_cutoff - frequency_cutoff)),
    ), dim=1)


def frequency_truncate(theta, new_freq_cutoff, strict=True):
    """Truncate a coefficient tensor to ``new_freq_cutoff``.

    When ``strict`` is true, reject a cutoff above the current cutoff.
    """
    frequency_cutoff = (theta.shape[1] - 1) // 2
    num_variables = theta.shape[0]

    if strict and new_freq_cutoff > frequency_cutoff:
        raise Exception('Truncation cutoff above original cutoff')

    const = theta[:, 0:1]
    a_params = theta[:, 1:frequency_cutoff+1]
    b_params = theta[:, frequency_cutoff+1:]

    return torch.cat((
        const,
        a_params[:, :new_freq_cutoff],
        b_params[:, :new_freq_cutoff],
    ), dim=1)


def frequency_multiply(theta, factor: int):
    """Increase the cutoff by ``factor`` while preserving existing modes."""
    frequency_cutoff = (theta.shape[1] - 1) // 2
    return frequency_extend(theta, frequency_cutoff * factor)
