git commands to resolve issue in pushing secerts

pip install git-filter-repo

git filter-repo --path chatApp/chains/retrievel_chain.py --invert-paths --force
git reflog expire --expire=now --all
git gc --prune=no

git remote add graphs https://github.com/rasmikrtp/Teams_MOL.git
git push graphs --force
