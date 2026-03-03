eps_clip=0.2

def ppo_update(old_probs, new_probs, advantage):
    ratio=new_probs/old_probs
    s1=ratio*advantage
    s2=torch.clamp(ratio,1-eps_clip,1+eps_clip)*advantage
    return -torch.min(s1,s2).mean()
