# Make sure to install twine: $pip install twine

import glob

builds = glob.glob('*/dist/*')

# Username: __token__
# Password: <see token on PyPi>

# Can change token via account settings on PyPi
# https://pypi.org/manage/account/#api-tokens

# 1. Remove previous token there: (a) scroll to API tokens, (b) Options -> "Remove token"
# 2. Then "Add API token"
# 3. Then use that as the token

# Also have to iterate version in setup.py, re-build via build.py, and delete previous distribution
# Alternatively, can delete previous package. "Your Projects" -> Select project -> Manage -> "Options" -> "Delete"
# Either way, first have to re-build via build.py
# MAKE SURE TO DELETE PRIOR dist/ AND build/ FOLDERS FIRST.

for build in builds:
    print(f'python -m twine upload {build}')

print('\nUsername: __token__'
      '\nPassword: <see token on PyPi>')

# ------------------------------------------------------------

# Undoing a push to GitHub:

# Note: A push to GitHub can be undone in Intellij Idea via the bottom "Git" tab

# -> right click the desired commit
# -> "Reset Current Branch to Here" -> Select "Hard" -> "Reset" -> Then "Git" in the window options at the top -> "Push"
# -> Click the arrow next to "Push" -> "Force Push"
# In order for the "Force Push" option to appear, might have to change some Intellij Idea Settings. Can look it up.
