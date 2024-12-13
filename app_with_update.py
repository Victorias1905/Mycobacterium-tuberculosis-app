import openai
import streamlit as st
import json
from pdfminer.high_level import extract_text
from transformers import AutoTokenizer, AutoModelForCausalLM
import unicodedata
import tiktoken
import os
import subprocess

def push_to_git_debug(model_name):
    repo_path = os.path.abspath(".")

    # Debug: Print current working directory and confirm .git presence
    st.write(f"Current working directory: {repo_path}")
    if not os.path.exists(os.path.join(repo_path, ".git")):
        st.error("No .git directory found. Please ensure this is a Git repository.")
        return

    # Check git installation and version
    git_version = subprocess.run(["git", "--version"], cwd=repo_path, capture_output=True, text=True)
    if git_version.returncode != 0:
        st.error("Git is not installed or not available in the PATH.")
        st.write("Git stdout:", git_version.stdout)
        st.write("Git stderr:", git_version.stderr)
        return
    else:
        st.write("Git version info:", git_version.stdout.strip())

    # Print out secrets structure for debugging (avoid printing sensitive info in production)
    # Note: Only do this if you're sure it's safe.
    st.write("st.secrets structure:", dict(st.secrets))

    # Retrieve GitHub token
    token = st.secrets.get("general", {}).get("GITHUB_TOKEN", None)
    if not token:
        st.error("GitHub token is missing in secrets. Add it to [general] in secrets.toml.")
        return

    # Try verifying the Git repository
    try:
        verify_repo = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=repo_path, capture_output=True, text=True)
        st.write("Repo verification stdout:", verify_repo.stdout)
        st.write("Repo verification stderr:", verify_repo.stderr)
        if verify_repo.returncode != 0:
            st.error("Not inside a valid Git repository.")
            return
    except subprocess.CalledProcessError as e:
        st.error(f"Error verifying Git repo:\nstdout: {e.stdout or 'No output'}\nstderr: {e.stderr or 'Unknown error'}")
        return

    # Configure Git user
    try:
        user_name_result = subprocess.run(["git", "config", "user.name", "Victorias1905"], cwd=repo_path, capture_output=True, text=True)
        st.write("Set user.name stdout:", user_name_result.stdout)
        st.write("Set user.name stderr:", user_name_result.stderr)

        user_email_result = subprocess.run(["git", "config", "user.email", "102805197+Victorias1905@users.noreply.github.com"], cwd=repo_path, capture_output=True, text=True)
        st.write("Set user.email stdout:", user_email_result.stdout)
        st.write("Set user.email stderr:", user_email_result.stderr)
    except subprocess.CalledProcessError as e:
        st.error(f"Error setting Git config:\nstdout: {e.stdout}\nstderr: {e.stderr}")
        return

    # Create or update the latest_model.json file
    file_path = os.path.join(repo_path, "latest_model.json")
    try:
        with open(file_path, "w") as file:
            json.dump({"model_name": model_name}, file)
        st.write("latest_model.json file created/updated successfully.")
    except Exception as e:
        st.error(f"Error writing latest_model.json:\n{e}")
        return

    # Stage changes
    try:
        add_result = subprocess.run(["git", "add", "."], cwd=repo_path, capture_output=True, text=True)
        st.write("Git add stdout:", add_result.stdout)
        st.write("Git add stderr:", add_result.stderr)
        if add_result.returncode != 0:
            st.error("Git add failed.")
            return
    except subprocess.CalledProcessError as e:
        st.error(f"Git add failed:\nstdout: {e.stdout}\nstderr: {e.stderr}")
        return

    # Commit changes
    try:
        commit_result = subprocess.run(
            ["git", "commit", "-m", f"Debug: Update model name to {model_name}"],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        st.write("Git commit stdout:", commit_result.stdout)
        st.write("Git commit stderr:", commit_result.stderr)
        if commit_result.returncode != 0:
            st.error("Git commit failed.")
            return
    except subprocess.CalledProcessError as e:
        st.error(f"Git commit failed:\nstdout: {e.stdout}\nstderr: {e.stderr}")
        return

    # Set GitHub remote URL
    username = "Victorias1905"
    repo_name = "Mycobacterium-tuberculosis-app"
    auth_remote = f"https://{token}@github.com/{username}/{repo_name}.git"
    st.write(f"Setting remote URL to: {auth_remote[:50]}... (truncated for security)")
    try:
        remote_result = subprocess.run(["git", "remote", "set-url", "origin", auth_remote], cwd=repo_path, capture_output=True, text=True)
        st.write("Set remote stdout:", remote_result.stdout)
        st.write("Set remote stderr:", remote_result.stderr)
        if remote_result.returncode != 0:
            st.error("Failed to set remote URL.")
            return
    except subprocess.CalledProcessError as e:
        st.error(f"Setting remote URL failed:\nstdout: {e.stdout}\nstderr: {e.stderr}")
        return

    # Push changes to GitHub
    # Ensure you are on the correct branch ('main' or 'master')
    branch_name = "main"
    try:
        push_result = subprocess.run(["git", "push", "origin", branch_name], cwd=repo_path, capture_output=True, text=True)
        st.write("Git push stdout:", push_result.stdout)
        st.write("Git push stderr:", push_result.stderr)
        if push_result.returncode == 0:
            st.success("Changes successfully pushed to GitHub!")
        else:
            st.error(f"Git push failed:\nstdout: {push_result.stdout}\nstderr: {push_result.stderr}")
    except subprocess.CalledProcessError as e:
        st.error(f"Git push command failed:\nstdout: {e.stdout}\nstderr: {e.stderr}")
    except Exception as e:
        st.error(f"Unexpected error during git push: {e}")

# User inputs the model name
model_name = st.text_input("Enter model name:")
if st.button("Push to GitHub"):
    if model_name:
        push_to_git_debug(model_name)
    else:
        st.error("Please enter a model name.")



