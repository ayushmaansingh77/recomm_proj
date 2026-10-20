"""Theme configuration and styling injection.

Strict adherence to minimal dark UI:
- Near-black background (#0B0B0C)
- Single raised surface (#141415)
- Hairline borders (#232326)
- Primary text (#EDEDED), muted text (#8A8A8E)
- Single accent (#5E6AD2)
- Inter typography with tight letter spacing (-0.01em)
- Smooth scrolling (Lenis + CSS fallback)
- Flat surfaces, zero gradients, zero shadows
"""

import streamlit as st
import streamlit.components.v1 as components

LENIS_INJECTION = """
<script>
(function() {
    try {
        const pWin = window.parent;
        const pDoc = pWin.document;
        if (!pWin.__lenis_initialized) {
            pWin.__lenis_initialized = true;
            const script = pDoc.createElement('script');
            script.src = 'https://unpkg.com/lenis@1.1.18/dist/lenis.min.js';
            script.onload = () => {
                try {
                    const lenis = new pWin.Lenis({
                        duration: 1.1,
                        easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
                        smoothWheel: true,
                        smoothTouch: false
                    });
                    function raf(time) {
                        lenis.raf(time);
                        pWin.requestAnimationFrame(raf);
                    }
                    pWin.requestAnimationFrame(raf);
                } catch(err) {
                    console.log('Lenis init error:', err);
                }
            };
            pDoc.head.appendChild(script);
        }
    } catch(e) {
        console.log('Smooth scrolling fallback active');
    }
})();
</script>
"""

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

/* Base reset & typography */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    background-color: #0B0B0C !important;
    color: #EDEDED !important;
    font-size: 15px !important;
    letter-spacing: -0.01em;
    scroll-behavior: smooth;
}

/* Hide header decorative bar */
header[data-testid="stHeader"] {
    background: transparent !important;
}

/* Remove default extra padding */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 3.5rem !important;
    max-width: 1220px !important;
}

/* Typography scale - calibrated for optimal legibility */
h1 {
    font-size: 1.85rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.025em !important;
    color: #EDEDED !important;
    margin-bottom: 0.35rem !important;
}

h2 {
    font-size: 1.35rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
    color: #EDEDED !important;
    margin-top: 1.25rem !important;
    margin-bottom: 0.5rem !important;
}

h3 {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.015em !important;
    color: #EDEDED !important;
    margin-bottom: 0.4rem !important;
}

h4 {
    font-size: 0.975rem !important;
    font-weight: 500 !important;
    letter-spacing: -0.01em !important;
    color: #EDEDED !important;
    margin-bottom: 0.35rem !important;
}

p, span, label {
    font-size: 0.9375rem !important;
    line-height: 1.6 !important;
    color: #EDEDED;
}

.stCaption, [data-testid="stCaptionContainer"] {
    color: #9B9BA1 !important;
    font-size: 0.875rem !important;
    line-height: 1.55 !important;
}

/* Flat containers and bordered blocks */
[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #141415 !important;
    border: 1px solid #232326 !important;
    border-radius: 6px !important;
    box-shadow: none !important;
    padding: 0.65rem !important;
}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 1.75rem;
    background-color: transparent !important;
    border-bottom: 1px solid #232326 !important;
    padding-bottom: 0;
}

.stTabs [data-baseweb="tab"] {
    background-color: transparent !important;
    border: none !important;
    color: #9B9BA1 !important;
    font-weight: 400 !important;
    font-size: 0.9375rem !important;
    padding: 0.6rem 0.25rem !important;
    border-bottom: 2px solid transparent !important;
}

.stTabs [aria-selected="true"] {
    color: #EDEDED !important;
    font-weight: 500 !important;
    border-bottom: 2px solid #5E6AD2 !important;
}

/* Inputs, textareas, selects */
input, textarea, select, [data-baseweb="select"] {
    background-color: #141415 !important;
    border: 1px solid #232326 !important;
    border-radius: 6px !important;
    color: #EDEDED !important;
    font-size: 0.875rem !important;
    box-shadow: none !important;
}

[data-baseweb="select"] > div {
    background-color: #141415 !important;
    border-color: #232326 !important;
    border-radius: 6px !important;
    color: #EDEDED !important;
}

input:focus, textarea:focus, [data-baseweb="select"]:focus-within > div {
    border-color: #5E6AD2 !important;
    outline: none !important;
    box-shadow: none !important;
}

/* Buttons */
button[kind="secondary"], button[data-testid="baseButton-secondary"] {
    background-color: #141415 !important;
    border: 1px solid #232326 !important;
    border-radius: 6px !important;
    color: #EDEDED !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    padding: 0.45rem 0.9rem !important;
    transition: border-color 0.15s ease, background-color 0.15s ease;
}

button[kind="secondary"]:hover, button[data-testid="baseButton-secondary"]:hover {
    border-color: #38383D !important;
    background-color: #18181A !important;
    color: #FFFFFF !important;
}

button[kind="primary"], button[data-testid="baseButton-primary"] {
    background-color: #5E6AD2 !important;
    border: 1px solid #5E6AD2 !important;
    border-radius: 6px !important;
    color: #FFFFFF !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    padding: 0.45rem 0.9rem !important;
}

button[kind="primary"]:hover, button[data-testid="baseButton-primary"]:hover {
    background-color: #4D58BF !important;
    border-color: #4D58BF !important;
}

/* Pills, tags, badges */
[data-baseweb="tag"] {
    background-color: #141415 !important;
    border: 1px solid #232326 !important;
    border-radius: 4px !important;
    color: #EDEDED !important;
    font-size: 0.8125rem !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background-color: #141415 !important;
    border: 1px solid #232326 !important;
    border-radius: 6px !important;
    padding: 0.85rem 1.15rem !important;
}

[data-testid="stMetricLabel"] {
    color: #9B9BA1 !important;
    font-size: 0.8125rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 500;
}

[data-testid="stMetricValue"] {
    color: #EDEDED !important;
    font-size: 1.45rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em;
}

/* Expanders */
[data-testid="stExpander"] {
    background-color: #141415 !important;
    border: 1px solid #232326 !important;
    border-radius: 6px !important;
}

[data-testid="stExpander"] details summary {
    font-size: 0.9375rem !important;
    font-weight: 500 !important;
    color: #9B9BA1 !important;
}

[data-testid="stExpander"] details summary:hover {
    color: #EDEDED !important;
}

/* Progress and sliders */
[data-baseweb="slider"] div[role="slider"] {
    background-color: #5E6AD2 !important;
    border: 2px solid #0B0B0C !important;
    box-shadow: none !important;
}

/* Custom UI Utility Classes */
.linear-card {
    background-color: #141415;
    border: 1px solid #232326;
    border-radius: 6px;
    padding: 1.1rem;
    margin-bottom: 0.85rem;
    display: flex;
    gap: 1.15rem;
    align-items: flex-start;
    transition: border-color 0.15s ease;
}

.linear-card:hover {
    border-color: #38383D;
}

.linear-badge {
    display: inline-block;
    font-size: 0.75rem;
    font-weight: 500;
    line-height: 1.1;
    padding: 0.28rem 0.55rem;
    border-radius: 4px;
    letter-spacing: 0.03em;
}

.badge-new {
    background-color: rgba(94, 106, 210, 0.15);
    color: #8E99F3;
    border: 1px solid rgba(94, 106, 210, 0.35);
}

.badge-neutral {
    background-color: #1C1C1E;
    color: #9B9BA1;
    border: 1px solid #28282B;
}

.badge-warm {
    background-color: #18181A;
    color: #EDEDED;
    border: 1px solid #28282B;
}

.stat-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    font-size: 0.875rem;
    color: #9B9BA1;
}

.stat-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background-color: #5E6AD2;
    display: inline-block;
}

.regime-banner {
    background-color: #141415;
    border: 1px solid #232326;
    border-left: 3px solid #5E6AD2;
    border-radius: 4px;
    padding: 0.9rem 1.15rem;
    margin-bottom: 1.1rem;
}

.regime-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #EDEDED;
    margin-bottom: 0.2rem;
}

.regime-desc {
    font-size: 0.875rem;
    color: #9B9BA1;
    line-height: 1.5;
}

/* Clean scrollbar */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #0B0B0C;
}
::-webkit-scrollbar-thumb {
    background: #232326;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
/* Radio navigation styled as sleek tabs */
.nav-tab-container div[data-testid="stRadio"] > div[role="radiogroup"] {
    display: flex !important;
    flex-direction: row !important;
    gap: 1.5rem !important;
    background-color: transparent !important;
    border-bottom: 1px solid #232326 !important;
    padding-bottom: 0 !important;
    margin-bottom: 1.25rem !important;
}

.nav-tab-container div[data-testid="stRadio"] > div[role="radiogroup"] label {
    background-color: transparent !important;
    border: none !important;
    color: #8A8A8E !important;
    font-weight: 400 !important;
    font-size: 0.875rem !important;
    padding: 0.5rem 0.25rem !important;
    border-bottom: 2px solid transparent !important;
    cursor: pointer !important;
    margin-bottom: -1px !important;
}

.nav-tab-container div[data-testid="stRadio"] > div[role="radiogroup"] label:hover {
    color: #EDEDED !important;
}

/* Hide the radio circle button */
.nav-tab-container div[data-testid="stRadio"] > div[role="radiogroup"] label > div:first-child {
    display: none !important;
}

/* Active tab style */
.nav-tab-container div[data-testid="stRadio"] > div[role="radiogroup"] label[data-checked="true"],
.nav-tab-container div[data-testid="stRadio"] > div[role="radiogroup"] label:has(input:checked) {
    color: #EDEDED !important;
    font-weight: 500 !important;
    border-bottom: 2px solid #5E6AD2 !important;
}
</style>
"""


def apply_theme():
    """Inject typography, custom styling, and Lenis smooth scrolling."""
    if hasattr(st, "html"):
        st.html(THEME_CSS)
        st.html(LENIS_INJECTION)
    else:
        st.markdown(THEME_CSS, unsafe_allow_html=True)
        components.html(LENIS_INJECTION, height=0, width=0)

