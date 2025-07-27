# --- Modified client.py with benchmark mode ---

import socket
import threading
import os
import time
import argparse
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

HOST = '127.0.0.1'
PORT = 12345


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
        return b""


def receive_messages(sock, aes_key):
    while True:
        try:
            encrypted_msg = sock.recv(2048)
            if encrypted_msg:
                decrypted_msg = aes_decrypt(aes_key, encrypted_msg)
                print(f"\r{decrypted_msg.decode()}\nYou> ", end="")
            else:
                break
        except:
            break


def benchmark_mode(sock, aes_key):
    print("[BENCHMARK] Starting 1MB AES encrypted round-trip test")
    test_data = os.urandom(1024 * 1024)  # 1 MB dummy data
    encrypted = aes_encrypt(aes_key, test_data)
    start = time.perf_counter()
    sock.send(encrypted)
    echoed = sock.recv(1024 * 1024 + 32)  # Allow room for IV + tag
    end = time.perf_counter()
    decrypted = aes_decrypt(aes_key, echoed)

    if decrypted == test_data:
        print(f"[BENCHMARK] Success. Round-trip time: {end - start:.6f} seconds")
    else:
        print("[BENCHMARK] Failed: Response data mismatch")


def main(benchmark=False):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((HOST, PORT))
        print("Connected to server.")

        server_public_key_pem = client_socket.recv(1024)
        server_public_key = serialization.load_pem_public_key(server_public_key_pem)
        aes_key = os.urandom(32)

        encrypted_key = server_public_key.encrypt(
            aes_key,
            padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()),
                         algorithm=hashes.SHA256(), label=None)
        )
        client_socket.send(encrypted_key)
        print("Secure session established.")

        if benchmark:
            benchmark_mode(client_socket, aes_key)
        else:
            threading.Thread(target=receive_messages, args=(client_socket, aes_key), daemon=True).start()
            while True:
                msg = input("You> ")
                if msg.lower() == 'exit':
                    break
                encrypted = aes_encrypt(aes_key, msg.encode())
                client_socket.send(encrypted)

    except Exception as e:
        print(f"Connection error: {e}")
    finally:
        client_socket.close()
        print("Disconnected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true", help="Run in benchmark mode")
    args = parser.parse_args()
    main(benchmark=args.benchmark)
