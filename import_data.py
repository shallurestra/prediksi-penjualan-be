"""Script untuk import data CSV ke PostgreSQL."""
import psycopg2
import psycopg2.extras
import csv

def parse_tanggal(raw):
    raw = str(raw).strip().strip('"').strip("'")
    parts = raw.replace("-", "/").split("/")
    if len(parts) == 3:
        if len(parts[2]) == 4:
            return f"{parts[2]}-{parts[1].zfill(2)}-{parts[0].zfill(2)}"
        elif len(parts[0]) == 4:
            return f"{parts[0]}-{parts[1].zfill(2)}-{parts[2].zfill(2)}"
    return raw

DAYS_MAP = {
    "senin": "Senin", "selasa": "Selasa", "rabu": "Rabu",
    "kamis": "Kamis", "jumat": "Jumat", "sabtu": "Sabtu", "minggu": "Minggu"
}

conn = psycopg2.connect(
    host="localhost", port=5432,
    dbname="postgres", user="postgres", password="shallu123"
)
cur = conn.cursor()

# Kosongkan dulu supaya tidak duplikat
cur.execute("DELETE FROM penjualan")
conn.commit()

rows = []
errors = 0

with open("data/dataset.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            tanggal = parse_tanggal(row["Tanggal"])
            hari_raw = row["Hari"].strip().strip('"').strip("'").lower()
            hari = DAYS_MAP.get(hari_raw, hari_raw.title())
            nama = row["Nama Produk"].strip().strip('"').strip("'")
            stok = int(str(row["Stok"]).strip().strip('"').strip("'"))
            terjual = int(str(row["Terjual"]).strip().strip('"').strip("'"))
            rows.append((tanggal, hari, nama, stok, terjual))
        except Exception as e:
            errors += 1

psycopg2.extras.execute_batch(
    cur,
    "INSERT INTO penjualan (tanggal, hari, nama_produk, stok, terjual) VALUES (%s, %s, %s, %s, %s)",
    rows,
    page_size=500
)
conn.commit()

cur.execute("SELECT COUNT(*) FROM penjualan")
total = cur.fetchone()[0]
conn.close()

print(f"Berhasil import : {total} baris")
print(f"Baris error     : {errors}")
