import distrax


def squashed_gaussian(mean, std):
    """diagonal Gaussian squashed through tanh, sample and log_prob both
    consistent with what actually gets executed in the env.

    the actor's `mean` here is the *pre-tanh* mean; `.sample()` returns
    values already in (-1, 1) and `.log_prob(actions)` expects post-tanh
    actions and applies the tanh Jacobian correction internally.
    """
    base = distrax.MultivariateNormalDiag(mean, std)
    return distrax.Transformed(base, distrax.Block(distrax.Tanh(), ndims=1))
