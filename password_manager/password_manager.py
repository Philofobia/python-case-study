from cryptography.fernet import Fernet, InvalidToken
import os

FOLDER = "password_manager"
KEY_FILE = os.path.join(FOLDER, "key.key")
PASSWORD_FILE = os.path.join(FOLDER, "passwords.txt")


def write_key():
    # Generates a key and save it into a file
    # wb is for write binary
    key = Fernet.generate_key()
    with open(KEY_FILE, "wb") as key_file:
        key_file.write(key)


def load_key():
    # Loads the key from the current directory
    # rb is for read binary
    return open(KEY_FILE, "rb").read()


# bytes is used to convert the string to bytes
#  key = load key but if it doesnt exist then write the key and key = load key
if not os.path.exists("key.key"):
    write_key()

key = load_key()
fer = Fernet(key)


def view():
    if not os.path.exists(PASSWORD_FILE):
        print("No passwords stored yet.")
        return
    # Open the file in read mode
    # r is for read, rstrip is to remove the new line character
    with open(PASSWORD_FILE, "r") as f:
        for line in f.readlines():
            data = line.rstrip()
            user, password = data.split(" | ")
            try:
                decrypted_password = fer.decrypt(password.encode()).decode()
                print("User:", user, "Password:", decrypted_password)
            except InvalidToken:
                print(f"Invalid token for user: {user}")


def add():
    user = input("Enter the username: ")
    password = input("Enter the password: ")

    # Open the file in append mode (creates the file if it doesn't exist)
    # a is for append
    with open(PASSWORD_FILE, "a") as f:
        encrypted_password = fer.encrypt(password.encode()).decode()
        f.write(user + " | " + encrypted_password + "\n")


# Main loop
while True:
    mode = input(
        "Would you like to add a new password or view existing ones (view, add), press q to quit? "
    ).lower()

    if mode == "q":
        break

    if mode == "view":
        view()
    elif mode == "add":
        add()
    else:
        print("Invalid mode.")
