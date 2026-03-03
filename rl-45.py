def kl_divergence(old,new):
    return (old*(old.log()-new.log())).sum()
