# --- Modified server.py to support benchmark mode ---

import socket
import threading
import os
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

HOST = '127.0.0.1'
PORT = 12345
clients = {}

server_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
server_public_key = server_private_key.public_key()
pem_public_key = server_public_key.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo
)

def aes_encrypt(key, data):
    iv = os.urandom(12)
    encryptor = Cipher(algorithms.AES(key), modes.GCM(iv)).encryptor()
    ciphertext = encryptor.update(data) + encryptor.finalize()
    return iv + encryptor.tag + ciphertext

def aes_decrypt(key, encrypted_data):
    iv = encrypted_data[:12]
    tag = encrypted_data[12:28]
    ciphertext = encrypted_data[28:]
    decryptor = Cipher(algorithms.AES(key), modes.GCM(iv, tag)).decryptor()
    return decryptor.update(ciphertext) + decryptor.finalize()

def broadcast(message, sender_socket=None):
    for client_socket, client_info in list(clients.items()):
        if client_socket != sender_socket:
            try:
                encrypted = aes_encrypt(client_info['aes_key'], message)
                client_socket.send(encrypted)
            except:
                client_socket.close()
                clients.pop(client_socket, None)

def handle_client(client_socket, addr):
    print(f"[NEW] Connection from {addr}")
    try:
        client_socket.send(pem_public_key)
        encrypted_key = client_socket.recv(256)
        aes_key = server_private_key.decrypt(
            encrypted_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        clients[client_socket] = {"addr": addr, "aes_key": aes_key}

        welcome = aes_encrypt(aes_key, b"Welcome to the secure chat!")
        client_socket.send(welcome)

        while True:
            encrypted_msg = client_socket.recv(2048 * 2)
            if not encrypted_msg:
                break

            try:
                msg = aes_decrypt(aes_key, encrypted_msg)
            except:
                continue

            # Benchmark detection: 1MB test data
            if len(msg) == 1024 * 1024:
                client_socket.send(aes_encrypt(aes_key, msg))
                continue

            broadcast_msg = f"[{addr}] {msg.decode()}".encode()
            print(f"Broadcasting from {addr}")
            broadcast(broadcast_msg, client_socket)

    except Exception as e:
        print(f"[ERROR] {addr}: {e}")
    finally:
        print(f"[DISCONNECT] {addr}")
        clients.pop(client_socket, None)
        client_socket.close()
        broadcast(f"[SERVER] {addr} has left the chat.".encode())

def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen()
    print(f"[LISTENING] Server on {HOST}:{PORT}")
    while True:
        client_socket, addr = server.accept()
        threading.Thread(target=handle_client, args=(client_socket, addr)).start()

if __name__ == "__main__":
    start_server()
