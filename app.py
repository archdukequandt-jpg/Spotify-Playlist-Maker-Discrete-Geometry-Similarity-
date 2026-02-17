
import os
import hashlib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

import plotly.express as px
import plotly.graph_objects as go

from data import load_csv, prepare_matrix
from geom import radar_points, raster_mask, overlap_uniqueness

st.set_page_config(page_title="PlaylistMaker (Discrete Geometry Matcher)", layout="wide")

# ---------------- Configuration ----------------
DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "Data-Table 1.csv")

# These columns are typical Spotify audio features.
DEFAULT_FEATURE_COLS = [
    "danceability", "energy", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "loudness"
]

META_COLS = ["track_id", "track_name", "artists", "album_name", "track_genre", "popularity", "duration_ms", "explicit"]

# Geometry raster settings (smaller grid is faster; 64 is a good default)
DEFAULT_GRID_N = 64
DEFAULT_SPAN = 1.08

def _safe_str(x):
    return "" if pd.isna(x) else str(x)

def _make_song_label(row: pd.Series) -> str:
    t = _safe_str(row.get("track_name", ""))
    a = _safe_str(row.get("artists", ""))
    g = _safe_str(row.get("track_genre", ""))
    # Keep labels short so selectbox is responsive
    label = f"{t} — {a}"
    tid = _safe_str(row.get("track_id",""))
    if tid:
        label += f"  ·  {tid[:8]}"
    if g:
        label += f"  ·  {g}"
    return label

@st.cache_data(show_spinner=False, max_entries=10)
def load_dataset(path: str) -> pd.DataFrame:
    df = load_csv(path)
    # Clean common artifact
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    # Ensure expected columns exist
    for c in ["track_name", "artists", "track_id"]:
        if c not in df.columns:
            raise ValueError(f"Dataset missing required column: '{c}'")
    # Make track_id string
    df["track_id"] = df["track_id"].astype(str)
    return df

@st.cache_data(show_spinner=False, max_entries=10)
def fit_scaler_and_nn(df: pd.DataFrame, feature_cols: list[str]):
    X = prepare_matrix(df, feature_cols)
    X = X.fillna(X.median(numeric_only=True))
    scaler = MinMaxScaler()
    Xs = scaler.fit_transform(X.values)
    # Nearest neighbors in feature space for a fast candidate shortlist
    nn = NearestNeighbors(n_neighbors=min(20000, len(df)), algorithm="auto", metric="euclidean")
    nn.fit(Xs)
    return scaler, nn, Xs

def compute_geometry_mask(values_01: np.ndarray, angles: np.ndarray, grid_n: int, span: float) -> np.ndarray:
    poly = radar_points(values_01.astype(np.float32), angles.astype(np.float32))
    return raster_mask(poly, grid_n=grid_n, span=span)

@st.cache_data(show_spinner=False, max_entries=200)
def compute_masks_for_indices(Xs: np.ndarray, indices: tuple[int, ...], grid_n: int, span: float) -> dict[int, np.ndarray]:
    """
    Compute discrete geometry masks for the given indices.
    Cached by Streamlit to avoid recomputation when toggling charts / parameters.
    """
    if len(indices) == 0:
        return {}
    n_feat = Xs.shape[1]
    angles = np.linspace(0, 2*np.pi, n_feat, endpoint=False).astype(np.float32)
    out = {}
    for i in indices:
        out[int(i)] = compute_geometry_mask(Xs[int(i)], angles, grid_n, span)
    return out

def rank_similar_songs(df: pd.DataFrame, Xs: np.ndarray, nn: NearestNeighbors, seed_idx: int,
                       playlist_size: int, candidate_pool: int, grid_n: int, span: float) -> pd.DataFrame:
    # 1) Candidate shortlist (fast)
    k = int(min(max(playlist_size*20, candidate_pool), len(df), 20000))
    distances, neighbors = nn.kneighbors(Xs[seed_idx].reshape(1, -1), n_neighbors=k, return_distance=True)
    neighbors = neighbors[0].astype(int).tolist()
    distances = distances[0].astype(float).tolist()

    # Remove the seed itself if present at the top
    if neighbors and neighbors[0] == seed_idx:
        neighbors = neighbors[1:]
        distances = distances[1:]

    # 2) Discrete geometry scoring on candidates
    idxs_for_masks = tuple([seed_idx] + neighbors[:candidate_pool])
    masks = compute_masks_for_indices(Xs, idxs_for_masks, grid_n=grid_n, span=span)

    seed_mask = masks.get(seed_idx)
    if seed_mask is None:
        raise RuntimeError("Failed to compute seed mask")

    scored = []
    for cand_idx, d in zip(neighbors[:candidate_pool], distances[:candidate_pool]):
        m = masks.get(int(cand_idx))
        if m is None:
            continue
        uniq = float(overlap_uniqueness(seed_mask, m))        # 0 = identical, 1 = disjoint
        sim = 1.0 - uniq                                      # Jaccard similarity
        scored.append((int(cand_idx), sim, uniq, float(d)))

    out = pd.DataFrame(scored, columns=["_idx", "similarity", "uniqueness", "feature_distance"])
    # Primary: geometry similarity; Secondary: feature distance
    out = out.sort_values(["similarity", "feature_distance"], ascending=[False, True]).head(int(playlist_size))
    meta_cols = [c for c in META_COLS if c in df.columns]
    out = out.merge(df.reset_index().rename(columns={"index": "_idx"})[["_idx"] + meta_cols], on="_idx", how="left")
    return out

# ---------------- Sidebar ----------------
st.sidebar.header("Data & settings")

data_mode = st.sidebar.radio("Dataset", ["Use included dataset", "Upload CSV"], index=0)

if data_mode == "Upload CSV":
    up = st.sidebar.file_uploader("Upload a Spotify-like CSV", type=["csv"])
    if up is None:
        st.info("Upload a CSV to continue, or switch back to the included dataset.")
        st.stop()
    dataset_path = up
else:
    dataset_path = DEFAULT_DATA_PATH

try:
    df = load_dataset(dataset_path)
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# Choose features (defaults are safe and interpretable)
numeric_candidates = [c for c in DEFAULT_FEATURE_COLS if c in df.columns]
extra_numeric = [c for c in ["popularity", "duration_ms"] if c in df.columns]
default_cols = numeric_candidates + extra_numeric

feature_cols = st.sidebar.multiselect(
    "Feature columns (scaled 0–1)",
    options=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])],
    default=default_cols
)

if len(feature_cols) < 3:
    st.warning("Pick at least 3 numeric features for the radar geometry.")
    st.stop()

playlist_size = st.sidebar.slider("Playlist size (top X)", 10, 200, 100, step=10)
candidate_pool = st.sidebar.slider("Candidate pool (fast prefilter)", 500, 20000, 5000, step=500)
grid_n = st.sidebar.select_slider("Geometry grid resolution", options=[32, 48, 64, 80, 96, 128], value=DEFAULT_GRID_N)
span = st.sidebar.slider("Geometry span", 1.02, 1.30, float(DEFAULT_SPAN), step=0.01)

st.sidebar.markdown("---")
st.sidebar.caption("Matching is done in two stages: (1) fast nearest-neighbor shortlist in feature space, then (2) discrete-geometry overlap (Jaccard) on radar-shape masks.")

# Fit scaler & NN
with st.spinner("Preparing feature matrix..."):
    scaler, nn, Xs = fit_scaler_and_nn(df, feature_cols)

# ---------------- Main UI ----------------
st.title("PlaylistMaker — Discrete Geometry Song Matcher")
st.write("Pick a seed song, then generate a playlist of the most similar tracks using discrete-geometry overlap between radar-shape masks.")

# ---------------- Instructions + credits ----------------
st.markdown("""### How to use this app
1. Choose your CSV in the left sidebar (default: **Data-Table 1.csv**).
2. Select the numeric feature columns (axes) that define each song’s geometry.
3. Use the search box to find a **seed song**, then pick it from the dropdown.
4. Choose the playlist size (**Top X**) and click **Generate playlist**.
5. Review the similarity chart, radar comparison, and embedding plot; export the playlist as CSV.

**Creators:** Ryan Childs (ryanchilds10@gmail.com) · James Quandt (archdukequandt@gmail.com) · James Belhund (jamesbelhund@gmail.com)
""")



# Search + pick seed song
left, right = st.columns([1.2, 1])

with left:
    query = st.text_input("Search song or artist", value="", help="Type part of a track name or artist to narrow the dropdown.")
    if query.strip():
        q = query.strip().lower()
        filt = df["track_name"].astype(str).str.lower().str.contains(q, na=False) | df["artists"].astype(str).str.lower().str.contains(q, na=False)
        cand = df.loc[filt].head(500).copy()
    else:
        cand = df.head(500).copy()

    cand["__label"] = cand.apply(_make_song_label, axis=1)
    options = cand["__label"].tolist()
    if not options:
        st.warning("No matches for that search. Try a different query.")
        st.stop()

    chosen_label = st.selectbox("Seed song", options=options, index=0)
    chosen_row = cand.loc[cand["__label"] == chosen_label].iloc[0]
    seed_idx = int(chosen_row.name)

with right:
    st.subheader("Seed song details")
    show_cols = [c for c in META_COLS if c in df.columns]
    st.dataframe(df.loc[[seed_idx], show_cols], use_container_width=True)

# Generate playlist
run = st.button("Generate playlist", type="primary")

if run:
    with st.spinner("Computing playlist (geometry overlap on candidates)..."):
        playlist = rank_similar_songs(df, Xs, nn, seed_idx, playlist_size, candidate_pool, grid_n, span)

    st.markdown("### Top matches")
    show_cols = [c for c in ["track_name","artists","album_name","track_genre","popularity","duration_ms","explicit","similarity","uniqueness","feature_distance"] if c in playlist.columns]
    st.dataframe(playlist[show_cols], use_container_width=True, height=420)

    # -------- Plotly charts --------
    st.markdown("### Similarity distribution (top matches)")
    bar_df = playlist.copy()
    bar_df["label"] = bar_df.apply(lambda r: f"{_safe_str(r.get('track_name'))} — {_safe_str(r.get('artists'))}", axis=1)
    fig_bar = px.bar(bar_df.sort_values("similarity", ascending=True),
                     x="similarity", y="label", orientation="h",
                     hover_data=["track_genre","popularity","feature_distance","uniqueness"])
    fig_bar.update_layout(height=650, margin=dict(l=20,r=20,t=30,b=20))
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("### Radar geometry: seed vs playlist average")
    # Build radar values (unscaled originals are less comparable; show scaled)
    seed_vals = Xs[seed_idx]
    mean_vals = Xs[playlist["_idx"].astype(int).values].mean(axis=0)

    feat = list(feature_cols)
    theta = feat + [feat[0]]
    seed_r = list(seed_vals) + [seed_vals[0]]
    mean_r = list(mean_vals) + [mean_vals[0]]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=seed_r, theta=theta, fill="toself", name="Seed (scaled)"))
    fig_radar.add_trace(go.Scatterpolar(r=mean_r, theta=theta, fill="toself", name="Playlist avg (scaled)"))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), height=520, margin=dict(l=20,r=20,t=30,b=20))
    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("### 2D embedding of seed + playlist")
    # PCA embedding on scaled features (fast + deterministic)
    emb_X = np.vstack([Xs[seed_idx], Xs[playlist["_idx"].astype(int).values]])
    pca = PCA(n_components=2, random_state=0)
    emb = pca.fit_transform(emb_X)
    emb_df = pd.DataFrame(emb, columns=["x","y"])
    emb_df["type"] = ["Seed"] + ["Playlist"] * (len(emb_df)-1)
    emb_df["similarity"] = [1.0] + playlist["similarity"].tolist()
    emb_df["track_name"] = [df.loc[seed_idx, "track_name"]] + playlist["track_name"].fillna("").tolist()
    emb_df["artists"] = [df.loc[seed_idx, "artists"]] + playlist["artists"].fillna("").tolist()
    fig_scatter = px.scatter(emb_df, x="x", y="y", color="type", size="similarity",
                             hover_data=["track_name","artists","similarity"])
    fig_scatter.update_layout(height=520, margin=dict(l=20,r=20,t=30,b=20))
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Download playlist
    st.markdown("### Export")
    csv_bytes = playlist.drop(columns=["_idx"]).to_csv(index=False).encode("utf-8")
    st.download_button("Download playlist CSV", data=csv_bytes, file_name="playlist_matches.csv", mime="text/csv")
else:
    st.info("Click **Generate playlist** to compute matches.")


st.markdown("---")

# ---------------- Discrete geometry + math explanation ----------------
st.markdown("## What the discrete-geometry scoring is doing (formulas + details)")

st.markdown("""### 1) Convert each song into a multi-axis shape (radar polygon)

Pick \(d\) numeric audio features. After normalization to \([0,1]\), each song becomes a vector of radii:

\[
\mathbf{v} = (v_0, v_1, \dots, v_{d-1}),\quad v_k \in [0,1]
\]

Angles are evenly spaced:

\[
\theta_k = \frac{2\pi k}{d}
\]

Polygon vertices in 2D:

\[
x_k = v_k \cos(\theta_k),\quad y_k = v_k \sin(\theta_k)
\]

This yields a radar / spider polygon for each song.
""")

st.markdown("""### 2) Discrete geometry: rasterize each polygon onto a grid

We create a uniform grid over \(([-s, s]\times[-s, s])\) with resolution \(N\times N\).
For each grid cell center \((x, y)\), we mark it as *inside* the polygon using a point-in-polygon test.
That produces a boolean mask \(M\) representing the polygon’s occupied area.
""")

st.markdown("""### 3) Similarity / uniqueness between two songs (Jaccard on masks)

Given two masks \(M_A\) and \(M_B\):

- Intersection area (overlap): \(|M_A \cap M_B|\)
- Union area: \(|M_A \cup M_B|\)

Jaccard overlap:

\[
J(A,B) = \frac{|M_A \cap M_B|}{|M_A \cup M_B|}
\]

In this app, we report **similarity** as \(J(A,B)\).  
(Equivalently, you can define a “uniqueness” score \(U(A,B) = 1 - J(A,B)\); higher \(U\) means less overlap.)
""")

st.markdown("""### 4) Playlist objective: maximize similarity to the seed

For a chosen seed song \(s\), each candidate song \(i\) receives a similarity:

\[
\text{sim}(s,i) = J(s,i)
\]

The playlist is the Top-\(X\) songs by \(\text{sim}(s,i)\) (excluding the seed itself), optionally with lightweight constraints (e.g., drop duplicates).
""")

st.markdown("""### 5) Optional neural acceleration (if enabled in other variants)

Computing exact overlap scores for many candidate comparisons can be expensive.
A small neural network can be trained to approximate similarity:

- **Input:** concatenated normalized features \([\mathbf{x}_s, \mathbf{x}_i]\)
- **Target:** exact \(J(s,i)\) from the discrete-geometry masks

The optimizer can then use \(\widehat{J}(s,i)\) for fast ranking, occasionally refreshing with exact mask scores.
""")

st.caption("Creators: Ryan Childs · James Quandt · James Belhund")
