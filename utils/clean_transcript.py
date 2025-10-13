import re

# from Commonvoice, what they are doing to the test transcripts.
bracketed = re.compile(r"\[[^\]]+\]")
unintell_paren = re.compile(r"\(\?+\)")
repl_punc = re.compile('[,?¿¡!";:]+')
multispace = re.compile("  +")
def clean(t):
    t = str(t)
    t = re.sub(bracketed, " ", t)
    t = re.sub(unintell_paren, " ", t)
    t = t.replace(" ... ", " ")
    t = t.replace("#x27;", "'")
    t = re.sub(repl_punc, " ", t)
    t = t.replace("...", "!ELLIPSIS!").replace(".", " ").replace("!ELLIPSIS!", "...")
    t = re.sub(multispace, " ", t)
    return t