# client.py

import socket
import threading
import os
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

# --- Client Configuration ---
HOST = '127.0.0.1'
PORT = 12345


# --- Helper Functions ---
def aes_encrypt(key, data):
    iv = os.urandom(12)
    encryptor = Cipher(algorithms.AES(key), modes.GCM(iv)).encryptor()
    ciphertext = encryptor.update(data) + encryptor.finalize()
    return iv + encryptor.tag + ciphertext


def aes_decrypt(key, encrypted_data):
    try:
        iv = encrypted_data[:12]
        tag = encrypted_data[12:28]
        ciphertext = encrypted_data[28:]
        decryptor = Cipher(algorithms.AES(key), modes.GCM(iv, tag)).decryptor()
        return decryptor.update(ciphertext) + decryptor.finalize()
    except Exception as e:
        print(f"[DECRYPTION ERROR] {e}")
        return b"Could not decrypt message."


def receive_messages(sock, aes_key, stop_event):
    """Listens for incoming messages from the server and decrypts them."""
    while not stop_event.is_set():
        try:
            encrypted_message = sock.recv(2048)
            if not encrypted_message:
                print("\n[INFO] Server closed the connection.")
                stop_event.set()
                break
            decrypted_message = aes_decrypt(aes_key, encrypted_message)
            print(f"\r{decrypted_message.decode('utf-8')}\nYou> ", end="")
        except Exception as e:
            print(f"\n[ERROR] Receiving failed: {e}")
            stop_event.set()
            break


def main():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    stop_event = threading.Event()

    try:
        client_socket.connect((HOST, PORT))
        print("Connected to server.")

        # --- Handshake ---
        server_public_key_pem = client_socket.recv(1024)
        server_public_key = serialization.load_pem_public_key(server_public_key_pem)

        aes_key = os.urandom(32)
        encrypted_aes_key = server_public_key.encrypt(
            aes_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        client_socket.send(encrypted_aes_key)
        print("Secure session established.")

        # --- Start receiving thread ---
        recv_thread = threading.Thread(target=receive_messages, args=(client_socket, aes_key, stop_event))
        recv_thread.start()

        # --- Main send loop ---
        while not stop_event.is_set():
            message = input("You> ")
            if message.lower() == 'exit':
                stop_event.set()
                break
            encrypted_message = aes_encrypt(aes_key, message.encode('utf-8'))
            try:
                client_socket.send(encrypted_message)
            except Exception as e:
                print(f"[SEND ERROR] {e}")
                stop_event.set()
                break

        # --- Graceful shutdown ---
        try:
            client_socket.shutdown(socket.SHUT_RDWR)
        except:
            pass
        recv_thread.join()

    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        client_socket.close()
        print("Disconnected from server.")


if __name__ == "__main__":
    main()
