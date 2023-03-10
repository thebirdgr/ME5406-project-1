# this file is for personal use, it is used for environment development. 
import yaml
import os
x = 0
pip = 0
with open("environment.yml") as f:
        y = yaml.safe_load(f)
        pip = y["dependencies"][-1]["pip"]
        x = y["dependencies"][:-1 or None]
        print(x)
s = ""
for r in x:
        l = r.split("=")
        s += l[0] + " " 

f = open("pip.txt", "w")
f.write(s)
f.close()