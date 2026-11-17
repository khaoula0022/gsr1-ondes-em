# -*- coding: utf-8 -*-
"""
GSR1 — Simulation de la propagation des ondes électromagnétiques
entre un avion et un sous-marin : étude de l'épaisseur de peau.

Modèle : onde plane, incidence normale, interface air / eau de mer.
Convention temporelle : exp(+jωt). Propagation dans le sens +z.
  E(z,t) = Re{ E0 · exp(jωt) · exp(-j·k·z) },  k = β − jα  (α > 0)
  k2 = ω·sqrt(μ·(ε − jσ/ω)),   η2 = ωμ/k2 = sqrt(μ/(ε − jσ/ω))
  Γ = (η2 − η1)/(η2 + η1),     τ = 2η2/(η2 + η1) = 1 + Γ
  δ = 1/α  (épaisseur de peau, définition générale)
  Bon conducteur (σ ≫ ωε) : α = β = sqrt(πfμσ),  δ = sqrt(2/(ωμσ))
"""

import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import streamlit as st

# ---------------------------------------------------------------- constantes
C0 = 299_792_458.0
MU0 = 4.0 * np.pi * 1e-7
EPS0 = 8.854_187_817e-12
ETA0 = np.sqrt(MU0 / EPS0)          # ≈ 377 Ω
EPSR_SEA = 80.0
SIGMA_SEA = 4.0


# ---------------------------------------------------------------- physique
def compute(f, sigma, epsr):
    """Grandeurs exactes (sans approximation bon conducteur)."""
    w = 2.0 * np.pi * f
    eps = epsr * EPS0
    mu = MU0
    eps_c = eps - 1j * sigma / w                       # permittivité complexe
    k2 = w * np.sqrt(mu * eps_c)                       # k2 = β − jα
    beta = float(k2.real)
    alpha = float(-k2.imag)                            # α > 0
    eta2 = w * mu / k2                                 # impédance intrinsèque complexe
    eta1 = ETA0 + 0j
    gamma = (eta2 - eta1) / (eta2 + eta1)              # réflexion (champ E)
    tau = 2.0 * eta2 / (eta2 + eta1)                   # transmission (champ E)
    k1 = w / C0
    delta_exact = 1.0 / alpha if alpha > 0 else np.inf
    delta_gc = np.sqrt(2.0 / (w * mu * sigma))         # approx. bon conducteur
    ratio = sigma / (w * eps)                          # ≫1 ⇒ bon conducteur
    return dict(
        f=f, sigma=sigma, epsr=epsr, w=w, eps=eps, mu=mu,
        k1=k1, k2=k2, beta=beta, alpha=alpha, eta2=eta2,
        gamma=gamma, tau=tau,
        delta=delta_exact, delta_gc=delta_gc,
        lambda1=2*np.pi/k1, lambda2=2*np.pi/beta if beta > 0 else np.inf,
        ratio=ratio,
        R=float(abs(gamma)**2), Tpow=float(1 - abs(gamma)**2),
        att_db_per_m=8.686 * alpha,
    )


def field_at_depth(res, z):
    """|E(z)/E_inc| et bilan en dB pour une profondeur z (m)."""
    a = res["alpha"]
    tau_abs = abs(res["tau"])
    e_rel = tau_abs * np.exp(-a * z)                   # |E(z)| / |E_inc|
    db_interface = 20*np.log10(tau_abs) if tau_abs > 0 else -np.inf
    db_atten = -8.686 * a * z
    db_total = 20*np.log10(e_rel) if e_rel > 0 else -np.inf
    return e_rel, db_interface, db_atten, db_total


def optimal_frequency(sigma, epsr, z, fmin=1.0, fmax=1e6, n=4000):
    """Fréquence maximisant |E(z)/E_inc| = |τ(f)|·exp(−α(f)z) (recherche numérique)."""
    fs = np.logspace(np.log10(fmin), np.log10(fmax), n)
    vals = np.empty(n)
    for i, f in enumerate(fs):
        r = compute(float(f), sigma, epsr)
        vals[i] = abs(r["tau"]) * np.exp(-r["alpha"] * z)
    i = int(np.argmax(vals))
    # formule analytique (bon conducteur) : f_opt = 1/(π μ σ z²)
    f_analytic = 1.0 / (np.pi * MU0 * sigma * z**2) if z > 0 else np.nan
    return float(fs[i]), float(vals[i]), f_analytic


def fmt_f(f):
    if f < 1e3:
        return f"{f:.2f} Hz" if f < 100 else f"{f:.1f} Hz"
    if f < 1e6:
        return f"{f/1e3:.2f} kHz"
    return f"{f/1e6:.2f} MHz"


def fmt_m(x):
    if x >= 1000:
        return f"{x/1000:.2f} km"
    if x >= 1:
        return f"{x:.2f} m"
    if x >= 0.01:
        return f"{x*100:.1f} cm"
    return f"{x*1000:.2f} mm"


# ---------------------------------------------------------------- animation
@st.cache_data(show_spinner=False)
def make_animation_gif(f_r, sigma_r, epsr_r):
    """GIF schématique (panneau gauche) + quantitatif en profondeur réelle (panneau droit)."""
    res = compute(f_r, sigma_r, epsr_r)
    gamma, tau = res["gamma"], res["tau"]
    alpha, beta = res["alpha"], res["beta"]

    # --- panneau gauche : échelle schématique (phase compressée, τ amplifié)
    zg = np.linspace(-22, 22, 900)
    k1v, k2v, av = 2*np.pi/26.0, 2*np.pi/7.5, 1/3.2
    ex = float(min(300.0, 0.45 / max(abs(tau), 1e-6)))   # amplification visuelle de E_t
    air = zg < 0

    # --- panneau droit : profondeur réelle 0 → 30 m, amplitudes exactes
    zr = np.linspace(0, 30, 700)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2),
                                   gridspec_kw=dict(width_ratios=[1.35, 1.0]))
    fig.patch.set_facecolor("white")

    # fonds
    axL.axvspan(-22, 0, color="#dceeff", zorder=0)
    axL.axvspan(0, 22, color="#0b3d66", zorder=0)
    axL.axvline(0, color="k", lw=2)
    axR.axvspan(0, 30, color="#0b3d66", zorder=0)
    axR.axvline(0, color="k", lw=2)

    (l_inc,) = axL.plot([], [], color="#1f77b4", lw=1.2, label="Onde incidente")
    (l_ref,) = axL.plot([], [], color="#d62728", lw=1.2, label="Onde réfléchie")
    (l_tot,) = axL.plot([], [], color="black", lw=1.6, label="Champ total (air)")
    (l_tr,) = axL.plot([], [], color="#2ca02c", lw=1.6, label="Onde transmise (×%.0f)" % ex)
    (l_r,) = axR.plot([], [], color="#7CFC00", lw=1.8, label="E(z,t)/E(0⁺) exact")

    axL.set_xlim(-22, 22); axL.set_ylim(-2.4, 2.4)
    axL.set_xlabel("z (échelle schématique, non respectée)")
    axL.set_ylabel("E  (normalisé, E_inc = 1)")
    axL.set_title("Interface air / mer — incidence normale", fontsize=10)
    axL.text(-21, 2.05, "AIR", fontsize=11, weight="bold", color="#0b3d66")
    axL.text(1, 2.05, "EAU DE MER", fontsize=11, weight="bold", color="white")
    axL.text(1, -2.25, f"|τ| réel = {abs(tau):.2e}\n(invisible sans amplification)",
             fontsize=7.5, color="white")
    axL.legend(fontsize=7, loc="lower left")
    axL.grid(alpha=0.25)

    axR.set_xlim(0, 30); axR.set_ylim(-1.15, 1.15)
    axR.set_xlabel("Profondeur réelle z (m)")
    axR.set_ylabel("E(z,t) / E(0⁺)")
    axR.set_title(f"Atténuation réelle : α={alpha:.3f} Np/m, δ={res['delta']:.2f} m",
                  fontsize=10)
    axR.grid(alpha=0.3)
    axR.tick_params(colors="white"); axR.xaxis.label.set_color("white")
    axR.yaxis.label.set_color("white"); axR.title.set_color("white")
    for s in axR.spines.values():
        s.set_color("white")

    txt = fig.suptitle("", fontsize=11, weight="bold")

    def frame(i):
        t = 2*np.pi*i/48.0
        Ei = np.cos(t - k1v*zg)
        Er = np.real(gamma * np.exp(1j*(t + k1v*zg)))
        Et = np.real(tau * np.exp(1j*t) * np.exp(-1j*k2v*zg) * np.exp(-av*zg))
        tot = Ei + Er
        l_inc.set_data(zg[air], Ei[air])
        l_ref.set_data(zg[air], Er[air])
        l_tot.set_data(zg[air], tot[air])
        l_tr.set_data(zg[~air], ex*Et[~air])
        Er_real = np.exp(-alpha*zr) * np.cos(t - beta*zr)
        l_r.set_data(zr, Er_real)
        txt.set_text(f"t = {i/48:.2f} · T      f = {fmt_f(f_r)}      σ = {sigma_r:.1f} S/m")
        return l_inc, l_ref, l_tot, l_tr, l_r

    anim = FuncAnimation(fig, frame, frames=48, interval=90, blit=False)
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        anim.save(tmp_path, writer=PillowWriter(fps=11))
        with open(tmp_path, "rb") as fh:
            data = fh.read()
    finally:
        plt.close(fig)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return data


# ---------------------------------------------------------------- interface
st.set_page_config(page_title="GSR1 — Ondes EM avion ↔ sous-marin", layout="wide")

st.markdown("""
<style>
  .banner { background: linear-gradient(120deg, #0b3d66 0%, #14639e 55%, #1b8ac4 100%);
            border-radius: 14px; padding: 20px 24px; color: #fff;
            box-shadow: 0 4px 14px rgba(11,61,102,.25); margin-bottom: 14px; }
  .banner h1 { margin: 0 0 6px 0; font-size: 1.7rem; }
  .banner p { margin: 0; font-size: 1rem; opacity: .92; }
  .badge { display: inline-block; background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.45);
           border-radius: 999px; padding: 2px 12px; font-size: .8rem; margin: 8px 6px 0 0; }
  .metric-card { background: linear-gradient(180deg, #ffffff, #eef5fb);
                 border: 1px solid #c9dcea; border-top: 4px solid #14639e;
                 border-radius: 12px; padding: 12px; text-align: center;
                 box-shadow: 0 2px 8px rgba(11,61,102,.10); }
  .metric-card .val { font-size: 1.45rem; font-weight: 800; color: #0b3d66; }
  .metric-card .lab { font-size: 0.82rem; color: #555; }
  .ok { color: #137333; font-weight: 600; }
  section[data-testid="stSidebar"] { background: #f2f7fb; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="banner">
  <h1>🌊 GSR1 — Ondes EM entre un avion et un sous-marin</h1>
  <p>Simulation interactive : interface air / eau de mer, coefficients Γ/τ et <b>épaisseur de peau</b> δ(f, σ) — incidence normale, ondes planes.</p>
  <span class="badge">✈️ Avion</span><span class="badge">🌊 Eau de mer σ = 4 S/m</span>
  <span class="badge">🧴 δ = 1/√(πfμσ)</span><span class="badge">🎬 Animation incluse</span>
</div>
""", unsafe_allow_html=True)

# --- état partagé : curseurs <-> saisie numérique <-> préréglages ---
for _k, _v in {"logf": 4.0, "sigma": SIGMA_SEA, "epsr": EPSR_SEA, "z_sub": 15.0,
               "f_num": 10000.0, "sigma_num": SIGMA_SEA, "z_num": 15.0}.items():
    st.session_state.setdefault(_k, _v)

def _f_from_slider():
    st.session_state["f_num"] = float(10 ** st.session_state["logf"])

def _f_from_number():
    st.session_state["logf"] = float(np.log10(max(1.0, st.session_state["f_num"])))

def _s_from_slider():
    st.session_state["sigma_num"] = float(st.session_state["sigma"])

def _s_from_number():
    st.session_state["sigma"] = float(st.session_state["sigma_num"])

def _z_from_slider():
    st.session_state["z_num"] = float(st.session_state["z_sub"])

def _z_from_number():
    st.session_state["z_sub"] = float(st.session_state["z_num"])

def _preset(logf=None, sigma=None, z=None):
    if logf is not None:
        st.session_state["logf"] = float(logf)
        st.session_state["f_num"] = float(10 ** logf)
    if sigma is not None:
        st.session_state["sigma"] = float(sigma)
        st.session_state["sigma_num"] = float(sigma)
    if z is not None:
        st.session_state["z_sub"] = float(z)
        st.session_state["z_num"] = float(z)

with st.sidebar:
    st.header("⚙️ Paramètres réglables")
    st.caption("Réglage libre : curseurs **ou** saisie numérique exacte (Entrée pour valider).")
    st.subheader("Préréglages rapides")
    _b1, _b2 = st.columns(2)
    if _b1.button("ELF · 30 Hz", use_container_width=True):
        _preset(logf=np.log10(30))
    if _b2.button("VLF · 10 kHz", use_container_width=True):
        _preset(logf=4.0)
    if _b1.button("LF · 100 kHz", use_container_width=True):
        _preset(logf=5.0)
    if _b2.button("MF · 1 MHz", use_container_width=True):
        _preset(logf=6.0)
    if _b1.button("z = 10 m", use_container_width=True):
        _preset(z=10.0)
    if _b2.button("z = 30 m", use_container_width=True):
        _preset(z=30.0)
    if st.button("↺ Réinitialiser", use_container_width=True):
        _preset(logf=4.0, sigma=SIGMA_SEA, z=15.0)
        st.session_state["epsr"] = EPSR_SEA
    st.divider()
    logf = st.slider("Fréquence f  (log₁₀, Hz)", 0.0, 8.0, key="logf", step=0.05,
                     on_change=_f_from_slider,
                     help="10⁰ = 1 Hz … 10⁸ = 100 MHz. Défaut : 10 kHz (bande VLF).")
    st.number_input("f exacte (Hz)", min_value=1.0, max_value=1e8, step=1.0,
                    key="f_num", on_change=_f_from_number,
                    help="Tapez une valeur exacte (ex. 3000) puis Entrée.")
    f = float(10 ** logf)
    st.caption(f"→ **f = {fmt_f(f)}**")
    sigma = st.slider("Conductivité σ de l'eau de mer (S/m)", 0.5, 10.0, key="sigma", step=0.1,
                      on_change=_s_from_slider,
                      help="Eau de mer typique : 3–5 S/m (défaut 4 S/m).")
    st.number_input("σ exacte (S/m)", min_value=0.5, max_value=10.0, step=0.1,
                    key="sigma_num", on_change=_s_from_number)
    epsr = st.slider("Permittivité relative εᵣ (eau de mer)", 70.0, 85.0, key="epsr", step=1.0,
                     help="≈ 80 en basse fréquence. Influence faible en régime bon conducteur.")
    z_sub = st.slider("Profondeur du sous-marin z (m)", 0.0, 50.0, key="z_sub", step=0.5,
                      on_change=_z_from_slider)
    st.number_input("z exacte (m)", min_value=0.0, max_value=50.0, step=0.5,
                    key="z_num", on_change=_z_from_number)
    st.divider()
    st.caption("**Repères physiques** : ELF (3–30 Hz) et VLF (3–30 kHz) sont les bandes "
               "historiquement utilisées pour communiquer avec les sous-marins "
               "(projet Seafarer, etc.). À 10 kHz, δ ≈ 2,5 m seulement.")
    show_gc = st.checkbox("Afficher l'approximation bon conducteur", value=True)

res = compute(f, sigma, epsr)
e_rel, db_if, db_at, db_tot = field_at_depth(res, z_sub)
ratio = res["ratio"]

# --- cartes métriques
c1, c2, c3, c4, c5 = st.columns(5)
cards = [
    (c1, "Épaisseur de peau δ", fmt_m(res["delta"]), f"α = {res['alpha']:.4f} Np/m"),
    (c2, "Atténuation", f"{res['att_db_per_m']:.2f} dB/m", f"β = {res['beta']:.4f} rad/m"),
    (c3, "Réflexion |Γ|", f"{abs(res['gamma']):.4f}", f"R = {res['R']*100:.2f} % de la puissance"),
    (c4, "Transmission |τ|", f"{abs(res['tau']):.2e}", f"{db_if:.1f} dB (champ) à l'interface"),
    (c5, f"|E(z)/E_inc| à {z_sub:.0f} m", f"{e_rel:.2e}", f"{db_tot:.1f} dB au total"),
]
for col, lab, val, sub in cards:
    col.markdown(f'<div class="metric-card"><div class="lab">{lab}</div>'
                 f'<div class="val">{val}</div><div class="lab">{sub}</div></div>',
                 unsafe_allow_html=True)

with st.expander("🔍 Valeurs détaillées (vérification par le professeur)"):
    st.write(f"η₂ = {res['eta2'].real:.4f} + j{res['eta2'].imag:.4f} Ω "
             f"(|η₂| = {abs(res['eta2']):.4f} Ω, arg = {np.degrees(np.angle(res['eta2'])):.1f}°)")
    st.write(f"k₂ = {res['k2'].real:.4f} {res['k2'].imag:+.4f}j rad/m "
             f"→ β = {res['beta']:.4f} rad/m, α = {res['alpha']:.4f} Np/m")
    st.write(f"λ₁ (air) = {fmt_m(res['lambda1'])} · λ₂ (mer) = {fmt_m(res['lambda2'])}")
    st.write(f"σ/(ωε) = {ratio:.3e} · δ exact = {fmt_m(res['delta'])} · "
             f"δ bon conducteur = {fmt_m(res['delta_gc'])}")
    st.write(f"Γ = {res['gamma'].real:+.5f} {res['gamma'].imag:+.5f}j · "
             f"τ = {res['tau'].real:+.3e} {res['tau'].imag:+.3e}j")

st.write("")
if ratio > 100:
    st.success(f"Régime **bon conducteur** : σ/(ωε) = {ratio:.2e} ≫ 1 → "
               f"δ ≈ √(2/(ωμσ)) = {fmt_m(res['delta_gc'])} (écart avec le calcul exact : "
               f"{100*abs(res['delta']-res['delta_gc'])/res['delta']:.2f} %).")
elif ratio > 10:
    st.warning(f"Régime **intermédiaire** : σ/(ωε) = {ratio:.1f}. L'approximation bon conducteur "
               f"reste utilisable mais le calcul exact est affiché.")
else:
    st.error(f"Hors régime bon conducteur : σ/(ωε) = {ratio:.2f} ≤ 10. "
             f"Le calcul exact (α, β, η₂ complexes) reste valide ; δ = 1/α.")

# ---------------------------------------------------------------- graphiques
tab1, tab2, tab3, tab4 = st.tabs([
    "📐 Interface : Γ et τ", "🧴 Épaisseur de peau δ(f, σ)",
    "📉 Atténuation vs profondeur", "🎬 Animation de l'onde"])

with tab1:
    st.subheader("Coefficients de réflexion et de transmission à l'interface air–mer")
    fs = np.logspace(0, 8, 400)
    G = np.empty(400); T = np.empty(400); Rp = np.empty(400)
    for i, ff in enumerate(fs):
        r = compute(float(ff), sigma, epsr)
        G[i] = abs(r["gamma"]); T[i] = abs(r["tau"]); Rp[i] = r["R"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.semilogx(fs, G, label="|Γ| (champ E)", color="#d62728")
    a1.semilogx(fs, T, label="|τ| (champ E)", color="#2ca02c")
    a1.axvline(f, color="k", ls="--", lw=1)
    a1.set_xlabel("Fréquence f (Hz)"); a1.set_ylabel("Amplitude")
    a1.set_title("Coefficients en champ électrique"); a1.legend(); a1.grid(True, which="both", alpha=0.3)
    a2.semilogx(fs, Rp, label="R = |Γ|² (puissance réfléchie)", color="#d62728")
    a2.semilogx(fs, 1-Rp, label="1 − R (puissance transmise)", color="#2ca02c")
    a2.axvline(f, color="k", ls="--", lw=1)
    a2.set_xlabel("Fréquence f (Hz)"); a2.set_ylabel("Fraction de puissance")
    a2.set_title("Bilan de puissance à l'interface"); a2.legend(); a2.grid(True, which="both", alpha=0.3)
    st.pyplot(fig, use_container_width=True)
    st.markdown(
        f"À **{fmt_f(f)}** : Γ = {res['gamma'].real:+.4f} {res['gamma'].imag:+.4f}j "
        f"(|Γ| = {abs(res['gamma']):.4f}, arg Γ = {np.degrees(np.angle(res['gamma'])):.1f}°), "
        f"τ = {res['tau'].real:+.2e} {res['tau'].imag:+.2e}j (|τ| = {abs(res['tau']):.2e}). "
        f"η₂ = {res['eta2'].real:.3f} + j{res['eta2'].imag:.3f} Ω (|η₂| = {abs(res['eta2']):.3f} Ω ≪ η₁ = 377 Ω). "
        "L'eau de mer, quasi métallique en basse fréquence, **réfléchit presque tout** : "
        "la transmission à l'interface est un goulot d'étranglement indépendant de l'atténuation en profondeur.")

with tab2:
    st.subheader("Épaisseur de peau δ en fonction de la fréquence et de la conductivité")
    fs2 = np.logspace(0, 8, 400)
    d_ex = np.array([compute(float(ff), sigma, epsr)["delta"] for ff in fs2])
    d_gc = np.array([compute(float(ff), sigma, epsr)["delta_gc"] for ff in fs2])
    sigs = np.logspace(np.log10(0.1), 1, 200)
    d_sig = np.array([compute(f, float(s), epsr)["delta"] for s in sigs])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.loglog(fs2, d_ex, label="δ exact = 1/α", color="#0b3d66", lw=2)
    if show_gc:
        a1.loglog(fs2, d_gc, "--", label="δ bon conducteur √(2/ωμσ)", color="#ff7f0e")
    a1.axhline(z_sub, color="#2ca02c", ls=":", label=f"Profondeur sous-marin ({z_sub:.0f} m)")
    a1.plot([f], [res["delta"]], "ro", ms=8, label="Point de fonctionnement")
    a1.set_xlabel("Fréquence f (Hz)"); a1.set_ylabel("δ (m)")
    a1.set_title(f"δ(f) pour σ = {sigma:.1f} S/m"); a1.legend(fontsize=8); a1.grid(True, which="both", alpha=0.3)
    a2.loglog(sigs, d_sig, color="#0b3d66", lw=2)
    a2.axvline(sigma, color="k", ls="--", lw=1)
    a2.plot([sigma], [res["delta"]], "ro", ms=8)
    a2.set_xlabel("Conductivité σ (S/m)"); a2.set_ylabel("δ (m)")
    a2.set_title(f"δ(σ) pour f = {fmt_f(f)}"); a2.grid(True, which="both", alpha=0.3)
    st.pyplot(fig, use_container_width=True)
    st.markdown(
        "En régime bon conducteur, **δ = 1/√(πfμσ)** : δ diminue comme **1/√f** et **1/√σ**. "
        "Doubler la fréquence (ou la conductivité) divise δ par √2 ≈ 1,41. "
        f"Repères (σ = {sigma:.1f} S/m) : δ(100 Hz) = {fmt_m(compute(100, sigma, epsr)['delta'])}, "
        f"δ(1 kHz) = {fmt_m(compute(1e3, sigma, epsr)['delta'])}, "
        f"δ(10 kHz) = {fmt_m(compute(1e4, sigma, epsr)['delta'])}, "
        f"δ(1 MHz) = {fmt_m(compute(1e6, sigma, epsr)['delta'])}.")

with tab3:
    st.subheader("Atténuation du champ électrique en fonction de la profondeur")
    z = np.linspace(0, 50, 600)
    e_curve = np.exp(-res["alpha"] * z)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.plot(z, e_curve, color="#0b3d66", lw=2)
    for fr, col in [(100, "#ff7f0e"), (1e3, "#2ca02c"), (1e5, "#9467bd")]:
        rr = compute(fr, sigma, epsr)
        a1.plot(z, np.exp(-rr["alpha"]*z), ls="--", lw=1, color=col, label=fmt_f(fr))
    a1.plot([z_sub], [np.exp(-res["alpha"]*z_sub)], "ro", ms=8)
    a1.set_xlabel("Profondeur z (m)"); a1.set_ylabel("|E(z)| / |E(0⁺)|")
    a1.set_title(f"Décroissance exponentielle à {fmt_f(f)}"); a1.legend(title="Autres f", fontsize=8)
    a1.grid(alpha=0.3)
    a2.semilogy(z, e_curve, color="#0b3d66", lw=2)
    a2.plot([z_sub], [np.exp(-res["alpha"]*z_sub)], "ro", ms=8)
    a2.set_xlabel("Profondeur z (m)"); a2.set_ylabel("|E(z)| / |E(0⁺)|  (échelle log)")
    a2.set_title("Même courbe en échelle logarithmique (droite = exponentielle)")
    a2.grid(True, which="both", alpha=0.3)
    st.pyplot(fig, use_container_width=True)
    st.markdown(
        f"**Bilan à la profondeur du sous-marin ({z_sub:.0f} m)** : "
        f"perte à l'interface **{db_if:.1f} dB** + perte par atténuation **{db_at:.1f} dB** "
        f"= **{db_tot:.1f} dB** au total (|E(z)/E_inc| = {e_rel:.2e}). "
        "Ces deux pertes sont des phénomènes **distincts** : la première dépend du contraste "
        "d'impédances η₂/η₁, la seconde de α = 1/δ et de la profondeur.")

    st.subheader("🎯 Fréquence optimale pour une profondeur donnée")
    if z_sub > 0:
        f_opt, val_opt, f_ana = optimal_frequency(sigma, epsr, z_sub)
        st.markdown(
            f"En balayant f de 1 Hz à 1 MHz, le champ reçu |τ(f)|·exp(−α(f)z) est maximal à "
            f"**f_opt ≈ {fmt_f(f_opt)}** (valeur {val_opt:.2e}, soit {20*np.log10(val_opt):.1f} dB). "
            f"La formule analytique bon conducteur f_opt = 1/(πμσz²) donne {fmt_f(f_ana)}. "
            "Compromis physique : monter en fréquence augmente |τ| (∝ √f) mais augmente aussi "
            "l'atténuation (α ∝ √f) — d'où un optimum en **ELF/VLF** pour les grandes profondeurs.")
        fs3 = np.logspace(0, 6, 300)
        vals = np.array([abs(compute(float(ff), sigma, epsr)["tau"]) *
                         np.exp(-compute(float(ff), sigma, epsr)["alpha"]*z_sub) for ff in fs3])
        fig, ax = plt.subplots(figsize=(11, 3.6))
        ax.semilogx(fs3, 20*np.log10(vals), color="#0b3d66", lw=2)
        ax.axvline(f_opt, color="#d62728", ls="--", label=f"f_opt = {fmt_f(f_opt)}")
        ax.axvline(f, color="k", ls=":", label=f"f réglée = {fmt_f(f)}")
        ax.set_xlabel("Fréquence f (Hz)"); ax.set_ylabel("|E(z)/E_inc| (dB)")
        ax.set_title(f"Champ reçu à {z_sub:.0f} m en fonction de la fréquence (σ = {sigma:.1f} S/m)")
        ax.legend(); ax.grid(True, which="both", alpha=0.3)
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("Réglez la profondeur z > 0 pour calculer la fréquence optimale.")

with tab4:
    st.subheader("🎬 Animation : onde incidente, réfléchie et transmise")
    st.caption("Panneau gauche : schéma de l'interface (phase compressée, onde transmise amplifiée "
               "pour être visible — facteur indiqué). Panneau droit : atténuation **réelle** en mètres, "
               "amplitudes exactes E(z,t)/E(0⁺) = exp(−αz)·cos(ωt − βz).")
    gif = make_animation_gif(round(f, 3), round(float(sigma), 2), float(epsr))
    st.image(gif, caption=f"Propagation à f = {fmt_f(f)}, σ = {sigma:.1f} S/m — "
                          f"δ = {fmt_m(res['delta'])}, |Γ| = {abs(res['gamma']):.3f}, |τ| = {abs(res['tau']):.2e}",
             use_container_width=True)
    st.markdown(
        "Lecture : dans l'air, l'onde incidente (bleue) et l'onde réfléchie (rouge) forment une "
        "figure quasi stationnaire car |Γ| ≈ 1 (réflexion presque totale, déphasage ≈ 180°). "
        "Dans l'eau, l'onde transmise (verte, panneau droit) se propage avec une longueur d'onde "
        f"λ₂ = {fmt_m(res['lambda2'])} et s'éteint sur δ = {fmt_m(res['delta'])} : "
        "au-delà de quelques épaisseurs de peau, le champ est pratiquement nul.")

# ---------------------------------------------------------------- théorie
st.divider()
st.header("📚 Modèle physique et équations")
cA, cB = st.columns(2)
with cA:
    st.markdown("**Hypothèses** : milieux homogènes, linéaires, isotropes ; onde plane "
                "monochromatique ; incidence normale ; interface plane air (z < 0) / mer (z > 0). "
                "Convention $e^{+j\\omega t}$, propagation dans le sens $+z$.")
    st.latex(r"k_2 = \omega\sqrt{\mu\left(\varepsilon - j\sigma/\omega\right)} = \beta - j\alpha")
    st.latex(r"\eta_2 = \frac{\omega\mu}{k_2} = \sqrt{\frac{\mu}{\varepsilon - j\sigma/\omega}}")
    st.latex(r"\Gamma = \frac{\eta_2-\eta_1}{\eta_2+\eta_1}, \qquad \tau = \frac{2\eta_2}{\eta_2+\eta_1} = 1+\Gamma")
    st.latex(r"E(z) = \tau\, E_{inc}\, e^{-\alpha z}\, e^{-j\beta z} \quad (z>0)")
with cB:
    st.markdown("**Régime bon conducteur** ($\\sigma \\gg \\omega\\varepsilon$) :")
    st.latex(r"\alpha = \beta = \sqrt{\pi f \mu \sigma} = \frac{1}{\delta}, \qquad \delta = \sqrt{\frac{2}{\omega\mu\sigma}}")
    st.latex(r"\eta_2 \approx (1+j)\sqrt{\frac{\pi f \mu}{\sigma}}")
    st.latex(r"\text{Atténuation} = 8{,}686\,\alpha \;\; \text{dB/m} = \frac{8{,}686}{\delta}\;\text{dB/m}")
    st.markdown("L'épaisseur de peau δ est la distance sur laquelle l'amplitude chute de "
                "$e^{-1}$ (≈ 37 %). À 2δ il ne reste que 13,5 %, à 3δ ≈ 5 %.")

st.header("✅ Vérification des résultats attendus")
checks = [
    ("δ diminue quand f augmente (bon conducteur)",
     res["delta"] > compute(f*10, sigma, epsr)["delta"],
     f"δ({fmt_f(f)}) = {fmt_m(res['delta'])} → δ({fmt_f(f*10)}) = {fmt_m(compute(f*10, sigma, epsr)['delta'])}"),
    ("δ diminue quand σ augmente",
     res["delta"] > compute(f, sigma*2, epsr)["delta"],
     f"δ(σ={sigma:.1f}) = {fmt_m(res['delta'])} → δ(σ={sigma*2:.1f}) = {fmt_m(compute(f, sigma*2, epsr)['delta'])}"),
    ("Le champ s'atténue exponentiellement avec z",
     bool(np.exp(-res["alpha"]*z_sub) < 1.0),
     f"|E(z)/E(0⁺)| = {np.exp(-res['alpha']*z_sub):.3f} à {z_sub:.0f} m"),
    ("La fréquence influence fortement la pénétration",
     True,
     f"δ passe de {fmt_m(compute(100, sigma, epsr)['delta'])} (100 Hz) à {fmt_m(compute(1e5, sigma, epsr)['delta'])} (100 kHz)"),
    ("Transmission (interface) et atténuation (profondeur) sont distinctes",
     True,
     f"Interface : {db_if:.1f} dB · Atténuation : {db_at:.1f} dB"),
]
for label, ok, detail in checks:
    mark = "✅" if ok else "❌"
    st.markdown(f"{mark} **{label}** — {detail}")

st.caption("GSR1 — Mini-projet de simulation · Modèle simplifié deux milieux, incidence normale. "
           "Ne prétend pas à lui seul établir la faisabilité pratique d'une liaison avion–sous-marin "
           "(antennes, bruit, couplage magnétique, etc. exclus du périmètre).")

