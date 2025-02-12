import numpy as np


stack_position_r2frame = (-0.3614, 0.1927)
# corner of tale of ur5e_2, where I usually stack blocks for collection by the robot

workspace_x_lims_default = (-0.9, -0.54)
workspace_y_lims_default = (-1.0, -0.55)

goal_tower_position = [-0.45, -1.15]


def valid_position(x, y, block_positions, min_dist):
    """
    Check if the position x, y is valid, i.e. not too close to any of the block_positions
    """
    if x is None or y is None:
        return False
    for block_pos in block_positions:
        if np.linalg.norm(np.array(block_pos) - np.array([x, y])) < min_dist:
            return False
    return True


def sample_block_positions_uniform(n_blocks, workspace_x_lims=workspace_x_lims_default,
                                   workspace_y_lims=workspace_y_lims_default, min_dist=0.07):
    """
    sample n_blocks positions within the workspace limits, spaced at least by 0.05m in each axis
    """
    block_positions = []
    for i in range(n_blocks):
        x = None
        y = None
        while valid_position(x, y, block_positions, min_dist=min_dist) is False:
            x = np.random.uniform(*workspace_x_lims)
            y = np.random.uniform(*workspace_y_lims)
        block_positions.append([x, y])

    return block_positions


def sample_block_positions_from_dists(blocks_dist, min_dist=0.07):
    """
    sample n_blocks positions within the workspace limits, spaced at least by 0.05m in each axis
    """
    block_positions = []
    for b in blocks_dist:
        x = None
        y = None
        while valid_position(x, y, block_positions, min_dist=min_dist) is False:
            x, y = b.sample(1)[0]

        block_positions.append([x, y])

    return block_positions


def sample_block_positions_from_dists_vectorized(blocks_dist, n_samples, min_dist=0.07, k=2):
    """Sample n_samples different states using fully vectorized operations"""

    n_blocks = len(blocks_dist)
    # Sample k*n_samples positions for each block at once
    all_samples = [b.sample(k * n_samples) for b in blocks_dist]
    all_samples = np.array(all_samples)  # Shape: [n_blocks, k*n_samples, 2]

    # Reshape to have all samples for each position together
    samples_reshaped = np.transpose(all_samples, (1, 0, 2))  # Shape: [k*n_samples, n_blocks, 2]

    # Calculate all pairwise distances at once for all samples
    diffs = samples_reshaped[:, :, None, :] - samples_reshaped[:, None, :, :]
    # Shape: [k*n_samples, n_blocks, n_blocks, 2]

    distances = np.sqrt(np.sum(diffs ** 2, axis=3))  # Shape: [k*n_samples, n_blocks, n_blocks]

    # Set diagonal for each sample to large number
    distances[:, np.arange(n_blocks), np.arange(n_blocks)] = 999

    # Find valid configurations where all distances meet minimum
    valid_mask = np.all(distances >= min_dist, axis=(1, 2))
    valid_states = samples_reshaped[valid_mask]

    if len(valid_states) < n_samples:
        # Need to sample more
        remaining = sample_block_positions_from_dists_vectorized(blocks_dist,
                                                                 n_samples - len(valid_states),
                                                                 min_dist, k)
        valid_states = np.concatenate([valid_states, remaining])

    return valid_states[:n_samples]


def test_sample_block_positions_vectorized(blocks_dist):
    # Setup test data - let's say we have 4 blocks with normal distributions
    from scipy.stats import multivariate_normal
    n_blocks = 4
    n_samples = 50000
    min_dist = 0.07

    # Generate samples
    samples = sample_block_positions_from_dists_vectorized(blocks_dist, n_samples, min_dist)

    # Shape tests
    assert samples.shape == (
    n_samples, n_blocks, 2), f"Expected shape {(n_samples, n_blocks, 2)} but got {samples.shape}"

    # Validity tests
    for sample_idx in range(n_samples):
        state = samples[sample_idx]
        # Check distances between all pairs of blocks
        for i in range(n_blocks):
            for j in range(i + 1, n_blocks):
                dist = np.linalg.norm(state[i] - state[j])
                assert dist >= min_dist, f"Invalid distance {dist} between blocks {i} and {j} in sample {sample_idx}"

    print("All tests passed!")
    print(f"Generated {n_samples} valid states")
    print(f"Example state shape: {samples[0].shape}")
    print("Example state:\n", samples[0])
