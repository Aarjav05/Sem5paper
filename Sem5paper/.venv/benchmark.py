# benchmark.py

import time
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.asymmetric import rsa, ec, padding
from cryptography.hazmat.primitives import hashes, serialization


def benchmark_aes():
    """Benchmarks AES-256 GCM performance."""
    print("\n--- Benchmarking AES-256 (Symmetric) ---")

    # 1. Generate a 256-bit key and a 12-byte nonce
    start_time = time.perf_counter()
    key = os.urandom(32)  # 256 bits
    nonce = os.urandom(12)  # 96 bits for GCM
    key_gen_time = time.perf_counter() - start_time

    # 2. Prepare data (1MB of random data)
    data = os.urandom(1024 * 1024)  # 1 MB

    # 3. Encrypt
    encryptor = Cipher(algorithms.AES(key), modes.GCM(nonce)).encryptor()

    start_time = time.perf_counter()
    ciphertext = encryptor.update(data) + encryptor.finalize()
    encryption_time = time.perf_counter() - start_time

    # ---- FIX IS HERE ----
    # After encrypting, we must get the authentication tag
    tag = encryptor.tag

    # 4. Decrypt
    # When creating the decryptor, we must provide the nonce AND the tag
    decryptor = Cipher(algorithms.AES(key), modes.GCM(nonce, tag)).decryptor()

    start_time = time.perf_counter()
    decrypted_data = decryptor.update(ciphertext) + decryptor.finalize()
    decryption_time = time.perf_counter() - start_time

    # 5. Print results
    print(f"Key Generation Time: {key_gen_time:.6f} seconds")
    print(f"Encryption Time (1MB): {encryption_time:.6f} seconds")
    print(f"Decryption Time (1MB): {decryption_time:.6f} seconds")

    # Verify correctness
    assert data == decrypted_data


def benchmark_rsa():
    """Benchmarks RSA-2048 performance."""
    print("\n--- Benchmarking RSA-2048 (Asymmetric) ---")

    # 1. Generate a 2048-bit RSA key pair
    start_time = time.perf_counter()
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    public_key = private_key.public_key()
    key_gen_time = time.perf_counter() - start_time

    # 2. Prepare data (a short message, as RSA can't encrypt large data directly)
    # This is a key limitation demonstrated here. We encrypt a 256-bit "session key".
    message = os.urandom(32)

    # 3. Encrypt
    start_time = time.perf_counter()
    ciphertext = public_key.encrypt(
        message,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    encryption_time = time.perf_counter() - start_time

    # 4. Decrypt
    start_time = time.perf_counter()
    decrypted_message = private_key.decrypt(
        ciphertext,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    decryption_time = time.perf_counter() - start_time

    # 5. Print results
    print(f"Key Generation Time: {key_gen_time:.6f} seconds")
    print(f"Encryption Time (32 bytes): {encryption_time:.6f} seconds")
    print(f"Decryption Time (32 bytes): {decryption_time:.6f} seconds")

    # Note: We don't benchmark large data encryption as it's not RSA's intended use.
    # This benchmark specifically highlights its use for key exchange.
    print("Note: RSA is used for small data like keys, not large files.")

    # Verify correctness
    assert message == decrypted_message


def benchmark_ecc():
    """Benchmarks ECC (secp256r1) for a key exchange scenario."""
    print("\n--- Benchmarking ECC (secp256r1) Key Exchange (Asymmetric) ---")

    # 1. Generate two ECC key pairs (one for each party)
    start_time = time.perf_counter()
    private_key_a = ec.generate_private_key(ec.SECP256R1())
    private_key_b = ec.generate_private_key(ec.SECP256R1())
    key_gen_time = time.perf_counter() - start_time

    # 2. Perform the key exchange (ECDH)
    start_time = time.perf_counter()
    shared_key_a = private_key_a.exchange(ec.ECDH(), private_key_b.public_key())
    shared_key_b = private_key_b.exchange(ec.ECDH(), private_key_a.public_key())
    exchange_time = time.perf_counter() - start_time

    # Verify correctness: both parties should derive the same shared secret
    assert shared_key_a == shared_key_b

    # 3. Print results
    print(f"Key Generation Time (2 pairs): {key_gen_time:.6f} seconds")
    print(f"Shared Key Derivation Time: {exchange_time:.6f} seconds")
    print("Note: ECC is used for fast key agreement, not direct data encryption.")


if __name__ == "__main__":
    print("=" * 40)
    print(" Cryptographic Performance Benchmark")
    print("=" * 40)

    benchmark_aes()
    benchmark_rsa()
    benchmark_ecc()

    print("\n--- Conclusion ---")
    print("AES is orders of magnitude faster for encrypting/decrypting actual data.")
    print("RSA and ECC key generation and operations are significantly slower,")
    print("highlighting why they are used for initial handshakes, not continuous communication.")