import socket
import threading
import os
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

# --- Server Configuration ---
HOST = '127.0.0.1'
PORT = 12345
clients = {}  # socket -> {'addr': addr, 'aes_key': key, 'name': "Client n"}

# --- Client ID Assignment ---
client_id_counter = 1
client_id_lock = threading.Lock()

# --- Generate Server Keys ---
server_private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048
)
server_public_key = server_private_key.public_key()
pem_public_key = server_public_key.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo
)

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
        return b"[ERROR] Could not decrypt message."

def broadcast(message, sending_client=None):
    for client_socket, client_info in clients.items():
        if client_socket != sending_client:
            try:
                encrypted_msg = aes_encrypt(client_info['aes_key'], message)
                client_socket.send(encrypted_msg)
            except Exception as e:
                print(f"[ERROR] Could not send to {client_info['addr']}: {e}")
                client_socket.close()
                if client_socket in clients:
                    del clients[client_socket]

def handle_client(client_socket, addr):
    global client_id_counter
    print(f"[NEW CONNECTION] {addr} connected.")

    try:
        # --- Hybrid Key Exchange ---
        client_socket.send(pem_public_key)
        encrypted_aes_key = client_socket.recv(256)

        aes_key = server_private_key.decrypt(
            encrypted_aes_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

        # Assign name
        with client_id_lock:
            client_name = f"Client {client_id_counter}"
            client_id_counter += 1

        # Store client info
        clients[client_socket] = {
            'addr': addr,
            'aes_key': aes_key,
            'name': client_name
        }

        print(f"[SECURITY] AES key established with {client_name} ({addr})")

        # Notify others
        broadcast(f"[SERVER] {client_name} has joined the chat.".encode('utf-8'), client_socket)

        # Welcome message
        client_socket.send(aes_encrypt(aes_key, f"Welcome, {client_name}!".encode('utf-8')))

        # --- Chat Loop ---
        while True:
            encrypted_message = client_socket.recv(2048)
            if not encrypted_message:
                break

            decrypted_message = aes_decrypt(aes_key, encrypted_message)
            broadcast_message = f"[{client_name}] {decrypted_message.decode('utf-8')}".encode('utf-8')
            print(broadcast_message.decode('utf-8'))
            broadcast(broadcast_message, client_socket)

    except Exception as e:
        print(f"[ERROR] {addr}: {e}")

    finally:
        if client_socket in clients:
            client_name = clients[client_socket]['name']
            del clients[client_socket]
            broadcast(f"[SERVER] {client_name} has left the chat.".encode('utf-8'))
        client_socket.close()
        print(f"[DISCONNECTED] {addr} disconnected.")

def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(5)
    print(f"[LISTENING] Server is running on {HOST}:{PORT}")

    while True:
        client_socket, addr = server.accept()
        thread = threading.Thread(target=handle_client, args=(client_socket, addr))
        thread.start()

if __name__ == "__main__":
    start_server()
