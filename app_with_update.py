def push_to_git_debug(model_name):
    try:
        repo_path = os.path.abspath(".")

        # Verify the Git repository
        subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=repo_path, check=True, capture_output=True, text=True)

        # Configure Git user
        subprocess.run(["git", "config", "user.name", "Victorias1905"], cwd=repo_path, check=True)
        subprocess.run(["git", "config", "user.email", "102805197+Victorias1905@users.noreply.github.com"], cwd=repo_path, check=True)

        # Stage and commit changes
        file_path = os.path.join(repo_path, "latest_model.json")
        with open(file_path, "w") as file:
            json.dump({"model_name": model_name}, file)

        subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
        subprocess.run(["git", "commit", "-m", f"Debug: Update model name to {model_name}"], cwd=repo_path, check=True, capture_output=True, text=True)

        # Get the GitHub token
        token = st.secrets.get("general", {}).get("GITHUB_TOKEN", None)
        if not token:
            st.error("GitHub token is missing in secrets.")
            return

        # Set GitHub remote URL
        username = "Victorias1905"
        repo_name = "Mycobacterium-tuberculosis-app"
        auth_remote = f"https://{token}@github.com/{username}/{repo_name}.git"
        subprocess.run(["git", "remote", "set-url", "origin", auth_remote], cwd=repo_path, check=True)

        # Push changes to GitHub
        result = subprocess.run(["git", "push", "origin", "main"], cwd=repo_path, capture_output=True, text=True)
        if result.returncode == 0:
            st.success("Changes successfully pushed to GitHub!")
        else:
            st.error(f"Git push failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")

        # Debugging details
        st.write(f"Remote URL: {auth_remote}")
        st.write(f"Token from secrets: {token[:4]}***")
    except subprocess.CalledProcessError as e:
        st.error(f"Git command failed:\nstdout: {e.stdout or 'No output'}\nstderr: {e.stderr or 'Unknown error'}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

# Streamlit interface
st.title("Git Push Debug Tool")

# User inputs the model name
model_name = st.text_input("Enter model name:")
if st.button("Push to GitHub"):
    if model_name:
        push_to_git_debug(model_name)
    else:
        st.error("Please enter a model name.")



