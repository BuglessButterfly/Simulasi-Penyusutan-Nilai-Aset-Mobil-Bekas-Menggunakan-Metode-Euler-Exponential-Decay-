# README – Cara Menjalankan Aplikasi

1. Pastikan file berikut berada dalam satu folder:
   - `app.py`
   - `Euler_Decay.ipynb`
   - `public_cars.csv`
   - (opsional) `requirements.txt`

2. Install semua library yang dibutuhkan:
```bash
pip install streamlit pandas numpy plotly matplotlib
````

Jika memakai `requirements.txt`:
```bash
pip install -r requirements.txt
```

3. Jalankan aplikasi Streamlit:
```bash
streamlit run app.py
```

4. Aplikasi akan terbuka otomatis di browser pada alamat:
   `http://localhost:8501`
   Jika tidak terbuka otomatis, copy link dari terminal dan buka manual di browser.