import time
import os
import matplotlib.pyplot as plt
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.asymmetric import rsa, ec, padding
from cryptography.hazmat.primitives import hashes

# Store timings for plotting
timings = {
    "AES": {},
    "RSA": {},
    "ECC": {}
}


def benchmark_aes():
    key = os.urandom(32)
    nonce = os.urandom(12)
    data = os.urandom(1024 * 1024)

    t0 = time.perf_counter()
    key_gen_time = time.perf_counter() - t0

    encryptor = Cipher(algorithms.AES(key), modes.GCM(nonce)).encryptor()
    t0 = time.perf_counter()
    ciphertext = encryptor.update(data) + encryptor.finalize()
    encryption_time = time.perf_counter() - t0
    tag = encryptor.tag

    decryptor = Cipher(algorithms.AES(key), modes.GCM(nonce, tag)).decryptor()
    t0 = time.perf_counter()
    decrypted_data = decryptor.update(ciphertext) + decryptor.finalize()
    decryption_time = time.perf_counter() - t0

    assert decrypted_data == data

    timings["AES"] = {
        "Key Gen": key_gen_time,
        "Encryption": encryption_time,
        "Decryption": decryption_time
    }


def benchmark_rsa():
    t0 = time.perf_counter()
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()
    key_gen_time = time.perf_counter() - t0

    message = os.urandom(32)

    t0 = time.perf_counter()
    ciphertext = public_key.encrypt(
        message,
        padding.OAEP(
            mgf=padding.MGF1(hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    encryption_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    decrypted_message = private_key.decrypt(
        ciphertext,
        padding.OAEP(
            mgf=padding.MGF1(hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    decryption_time = time.perf_counter() - t0

    assert decrypted_message == message

    timings["RSA"] = {
        "Key Gen": key_gen_time,
        "Encryption": encryption_time,
        "Decryption": decryption_time
    }


def benchmark_ecc():
    t0 = time.perf_counter()
    private_key_a = ec.generate_private_key(ec.SECP256R1())
    private_key_b = ec.generate_private_key(ec.SECP256R1())
    key_gen_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    shared_key_a = private_key_a.exchange(ec.ECDH(), private_key_b.public_key())
    shared_key_b = private_key_b.exchange(ec.ECDH(), private_key_a.public_key())
    exchange_time = time.perf_counter() - t0

    assert shared_key_a == shared_key_b

    timings["ECC"] = {
        "Key Gen": key_gen_time,
        "Key Exchange": exchange_time,
        "Decryption": 0.0  # ECC not used for encryption here
    }


def plot_results():
    categories = ["Key Gen", "Encryption", "Decryption"]
    algorithms = list(timings.keys())

    x = range(len(categories))
    bar_width = 0.2

    fig, ax = plt.subplots()
    for i, algo in enumerate(algorithms):
        values = [timings[algo].get(cat, 0) for cat in categories]
        ax.bar([p + bar_width * i for p in x], values, bar_width, label=algo)

    ax.set_xlabel("Operation")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Cryptographic Operation Benchmark")
    ax.set_xticks([p + bar_width for p in x])
    ax.set_xticklabels(categories)
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    benchmark_aes()
    benchmark_rsa()
    benchmark_ecc()
    plot_results()
