import os
import json
import subprocess
import streamlit as st

def push_to_git(model_name):
    repo_path = os.path.abspath(".")

    # Write the latest model to file
    file_path = os.path.join(repo_path, "latest_model.json")
    with open(file_path, "w") as file:
        json.dump({"model_name": model_name}, file)

    # Configure Git user
    subprocess.run(["git", "config", "user.name", "Victorias1905"], cwd=repo_path, check=True)
    subprocess.run(["git", "config", "user.email", "102805197+Victorias1905@users.noreply.github.com"], cwd=repo_path, check=True)

    # Stage and commit changes
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
    subprocess.run(["git", "commit", "-m", f"Update model name to {model_name}"], cwd=repo_path, check=True)

    # Get the GitHub token from secrets
    token = st.secrets["general"]["GITHUB_TOKEN"]
    username = "Victorias1905"
    repo_name = "Mycobacterium-tuberculosis-app"
    auth_remote = f"https://{token}@github.com/{username}/{repo_name}.git"

    # Set GitHub remote URL
    subprocess.run(["git", "remote", "set-url", "origin", auth_remote], cwd=repo_path, check=True)

    # Push changes to GitHub
    result = subprocess.run(["git", "push", "origin", "main"], cwd=repo_path, capture_output=True, text=True)
    if result.returncode == 0:
        st.success("Changes successfully pushed to GitHub!")
    else:
        st.error(f"Git push failed:\n{result.stdout}\n{result.stderr}")

# Streamlit UI
model_name = st.text_input("Enter model name:")
if st.button("Push to GitHub"):
    if model_name:
        push_to_git(model_name)
    else:
        st.error("Please enter a model name.")



