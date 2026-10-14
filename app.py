"""
Know Your Scheme - A Government Scheme Eligibility Finder
Streamlit app  |  run:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analytics as A
import engine as E

st.set_page_config(page_title="Know Your Scheme", page_icon="🇮🇳", layout="wide")

GREEN, SAFFRON, NAVY = "#138808", "#FF9933", "#0B2E4F"

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
st.markdown(f"""
<style>
.block-container {{ padding-top: 1.4rem; max-width: 1250px; }}
.hero {{ background: linear-gradient(120deg, {NAVY} 0%, #0f5a46 60%, {GREEN} 100%);
        padding: 30px 36px; border-radius: 20px; color: #fff; margin-bottom: 18px;
        border-bottom: 6px solid {SAFFRON}; }}
.hero h1 {{ margin: 0; font-size: 2.3rem; color: #fff; }}
.hero p {{ margin: 6px 0 0 0; font-size: 1.05rem; opacity: .92; }}
.stat-strip {{ display:flex; gap:14px; flex-wrap:wrap; margin-top:18px; }}
.stat {{ background: rgba(255,255,255,.14); border-radius: 12px; padding: 10px 18px; min-width: 130px; }}
.stat b {{ font-size: 1.45rem; display:block; }}
.stat span {{ font-size: .78rem; opacity:.85; text-transform: uppercase; letter-spacing:.05em; }}
.chip {{ display:inline-block; padding: 2px 10px; border-radius: 99px; font-size: .75rem;
        margin-right: 6px; background:#EAF3EE; color:#0f5a46; font-weight:600; }}
.chip.o {{ background:#FFF1E0; color:#B25E00; }}
.chip.b {{ background:#E7EEF8; color:{NAVY}; }}
.badge {{ float:right; padding: 6px 14px; border-radius: 99px; color:#fff; font-weight:700; font-size:.95rem; }}
.hi {{ background:#0b7a2e; }} .mid {{ background:#2e9e4f; }} .lo {{ background:#d98a00; }}
.small {{ color:#5b6770; font-size:.85rem; }}
div[data-testid="stMetric"] {{ background:#F3F7F4; border-radius:14px; padding:12px 16px; border:1px solid #E1EAE4; }}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Helpers (work across old/new Streamlit width APIs)
# --------------------------------------------------------------------------- #
def show(fig, **kw):
    fig.update_layout(margin=dict(l=10, r=10, t=45, b=10), font=dict(size=13))
    try:
        st.plotly_chart(fig, width="stretch", **kw)
    except Exception:
        st.plotly_chart(fig, use_container_width=True, **kw)


def table(df, **kw):
    try:
        st.dataframe(df, hide_index=True, width="stretch", **kw)
    except Exception:
        st.dataframe(df, hide_index=True, use_container_width=True, **kw)


@st.cache_data(show_spinner=False)
def get_data():
    return E.load_schemes(), E.load_nfhs()


@st.cache_resource(show_spinner="Training eligibility models (one-time, ~10 s)…")
def get_models():
    return E.train_eligibility_models(E.load_schemes())


@st.cache_resource(show_spinner="Indexing scheme text…")
def get_text_index():
    return E.build_text_index(E.load_schemes())


@st.cache_data(show_spinner=False)
def get_sweep():
    return E.kmeans_sweep(E.load_nfhs())


@st.cache_data(show_spinner=False)
def get_clusters(k):
    return E.cluster_districts(E.load_nfhs(), k)


@st.cache_data(show_spinner="Running Random Forest regression…")
def get_drivers(target):
    return E.driver_analysis(E.load_nfhs(), target)


schemes, nfhs = get_data()
bundle = get_models()
tindex = get_text_index()
STATES = E.list_states(schemes)
rf_res, lr_res = bundle["results"][1], bundle["results"][0]
IND = E.indicator_cols(nfhs)


def pick(name):
    return IND.index(name) if name in IND else 0

# --------------------------------------------------------------------------- #
# Hero + sidebar
# --------------------------------------------------------------------------- #
st.markdown(f"""
<div class="hero">
  <h1>🇮🇳 Know Your Scheme</h1>
  <p>Find every Central &amp; State welfare scheme you are eligible for — in under a minute.</p>
  <div class="stat-strip">
    <div class="stat"><b>{len(schemes):,}</b><span>Schemes</span></div>
    <div class="stat"><b>{(schemes['level'] == 'Central').sum()}</b><span>Central</span></div>
    <div class="stat"><b>{(schemes['level'] == 'State').sum():,}</b><span>State / UT</span></div>
    <div class="stat"><b>{len(nfhs)}</b><span>Districts analysed</span></div>
    <div class="stat"><b>{rf_res['Accuracy']:.1%}</b><span>Model accuracy</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("About")
    st.write("A data-driven eligibility finder that matches your profile with government "
             "schemes, ranks them with a **match score**, and uses **NFHS-5** district data "
             "to show where schemes are needed most.")
    st.metric("Random Forest accuracy", f"{rf_res['Accuracy']:.1%}", f"{rf_res['Accuracy'] - lr_res['Accuracy']:+.1%} vs baseline")
    st.caption(f"Measured on {bundle['n_test']:,} unseen (citizen, scheme) pairs.")
    st.divider()
    st.caption("Pipeline: Data → Cleaning → EDA → Eligibility engine → Match score → "
               "ML (Logistic Regression · Random Forest · K-Means) → Deployment")
    st.caption("Information is indicative. Always confirm eligibility on the official scheme portal.")

tab_find, tab_eda, tab_dist, tab_work, tab_model, tab_about = st.tabs(
    ["🔍 Find Schemes", "📊 Data Insights (EDA)", "🗺️ District Analytics", "🧪 Analyze Your Data",
     "🤖 Model Evaluation", "ℹ️ About"])

# --------------------------------------------------------------------------- #
# TAB 1 - Find schemes
# --------------------------------------------------------------------------- #
def why_text(s, p):
    out = []
    if s["level"] == "State" and s["state"] not in E.NO_STATE:
        out.append(f"State scheme for your state ({s['state']})")
    if s["min_age"] > 0 or s["max_age"] < 120:
        hi = f"–{s['max_age']}" if s["max_age"] < 120 else "+"
        out.append(f"Your age ({p['age']}) fits the {s['min_age']}{hi} window")
    if s["gender"] not in ("", "All"):
        out.append(f"Open to {s['gender'].lower()} applicants")
    if s["caste_list"]:
        out.append("Category matches (" + " / ".join(s["caste_list"]) + ")")
    if s["occ_list"]:
        out.append("Occupation / situation matches (" + ", ".join(s["occ_list"]) + ")")
    if pd.notna(s["max_annual_income"]) and p.get("income") is not None:
        out.append(f"Income ₹{p['income']:,} is within the ₹{int(s['max_annual_income']):,} limit")
    if s["bpl_required"]:
        out.append("BPL household requirement met")
    return out or ["No restrictive criteria in the structured data — open to all"]


with tab_find:
    st.subheader("Tell us about yourself")
    with st.form("profile"):
        c1, c2, c3 = st.columns(3)
        state = c1.selectbox("State / UT", STATES, index=STATES.index("Tamil Nadu") if "Tamil Nadu" in STATES else 0)
        age = c2.number_input("Age", 5, 100, 18)
        gender = c3.selectbox("Gender", E.GENDERS)
        c4, c5, c6 = st.columns(3)
        income = c4.number_input("Annual family income (₹)", 0, 10_000_000, 120_000, step=10_000)
        skip_income = c4.checkbox("Prefer not to say (skip income rules)")
        category = c5.selectbox("Social category", E.CATEGORIES, index=1)
        occupation = c6.selectbox("Occupation / situation", E.OCCUPATIONS)
        k1, k2, k3, k4, k5 = st.columns(5)
        bpl = k1.checkbox("BPL household")
        disability = k2.checkbox("Person with disability")
        widow = k3.checkbox("Widow")
        minority = k4.checkbox("Minority community")
        ews = k5.checkbox("EWS")
        interests = st.multiselect("Areas of interest (optional)", sorted(schemes["category_tag"].unique()))
        query = st.text_input("Describe what you need (optional)", placeholder="e.g. scholarship for education")
        submitted = st.form_submit_button("🔎 Find my schemes", type="primary")

    if submitted:
        st.session_state["prof"] = dict(
            age=int(age), gender=gender, category=category, occupation=occupation,
            income=None if skip_income else int(income), bpl=bpl, disability=disability,
            widow=widow, minority=minority, ews=ews, state=state)
        st.session_state["interests"], st.session_state["query"] = interests, query

    p = st.session_state.get("prof")
    if not p:
        st.info("Fill in the form and press **Find my schemes** to see your personalised, ranked list.")
    else:
        res = E.recommend(p, schemes, st.session_state["interests"], st.session_state["query"], tindex)
        conf = E.ml_confidence(bundle, p, schemes)
        res["ml_conf"] = pd.Series(conf, index=schemes["slug"]).reindex(res["slug"]).values
        elig = res[res["eligible"]].copy()
        near = res[(~res["eligible"]) & (res["n_fails"] == 1)]
        agree = float((elig["ml_conf"] >= .5).mean()) if len(elig) else 0.0

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Eligible schemes", f"{len(elig):,} / {len(res):,}")
        m2.metric("Best match", f"{elig['match'].max():.0f}%" if len(elig) else "—")
        m3.metric("Your state's schemes", int((elig["level"] == "State").sum()))
        m4.metric("Almost eligible", len(near))
        m5.metric("ML agrees", f"{agree:.0%}")

        if elig.empty:
            st.warning("No scheme matches this profile exactly. Check the 'almost eligible' list below.")
        else:
            f1, f2, f3, f4 = st.columns([3, 1.5, 1.5, 1.5])
            cats = f1.multiselect("Filter by category", sorted(elig["category_tag"].unique()))
            lvl = f2.selectbox("Level", ["All", "Central", "State"])
            min_match = f3.slider("Minimum match %", 0, 99, 0)
            top_n = f4.slider("Show top", 10, 100, 25, step=5)
            view = elig[(elig["match"] >= min_match)]
            if cats:
                view = view[view["category_tag"].isin(cats)]
            if lvl != "All":
                view = view[view["level"] == lvl]
            view = view.head(top_n)

            left, right = st.columns([3, 1.4])
            with left:
                st.caption(f"Showing {len(view)} of {len(elig):,} eligible schemes, ranked by match score.")
                for i, (_, s) in enumerate(view.iterrows(), 1):
                    cls = "hi" if s["match"] >= 80 else "mid" if s["match"] >= 65 else "lo"
                    with st.container(border=True):
                        where = s["state"] if s["level"] == "State" else "All India"
                        st.markdown(
                            f"<span class='badge {cls}'>{s['match']:.0f}% match</span>"
                            f"**{i}. {s['scheme_name']}**<br>"
                            f"<span class='chip'>{s['category_tag']}</span>"
                            f"<span class='chip o'>{s['level']} · {where}</span>"
                            f"<span class='chip b'>AI confidence {s['ml_conf']:.0%}</span>",
                            unsafe_allow_html=True)
                        with st.expander("View details"):
                            st.markdown("**Why you match**")
                            for w in why_text(s, p):
                                st.markdown(f"- ✅ {w}")
                            if s["benefits"]:
                                st.markdown(f"**Benefits:** {s['benefits'][:700]}")
                            if s["eligibility"]:
                                st.markdown(f"**Official eligibility text:** {s['eligibility'][:900]}")
                            if s["documents"]:
                                st.markdown(f"**Documents required:** {s['documents'][:400]}")
                            if s["application"]:
                                st.markdown(f"**How to apply:** {s['application'][:500]}")
                            st.markdown(f"[Open on myScheme ↗]({s['official_link']})")
            with right:
                cat_counts = view["category_tag"].value_counts().reset_index()
                cat_counts.columns = ["Category", "Schemes"]
                show(px.pie(cat_counts, names="Category", values="Schemes", hole=.55,
                            title="Your matches by category",
                            color_discrete_sequence=px.colors.qualitative.Safe))
                out = elig[["scheme_name", "category_tag", "level", "state", "match", "benefits", "eligibility",
                            "documents", "official_link"]]
                st.download_button(f"⬇️ Download all {len(elig):,} matches (CSV)", out.to_csv(index=False).encode(),
                                   "my_schemes.csv", "text/csv")
                st.caption("The structured filters cover age, gender, income, category, occupation and state. "
                           "Always read the official eligibility text before applying.")

        if len(near):
            with st.expander(f"🟡 Almost eligible — {len(near):,} scheme(s) missing just one condition"):
                for _, s in near.head(15).iterrows():
                    st.markdown(f"**{s['scheme_name']}** — {s['fails'][0]}  \n"
                                f"<span class='small'>{s['benefits'][:160]}</span>", unsafe_allow_html=True)

        if p["state"] in set(nfhs["State"]):
            with st.expander("📍 Know your district (NFHS-5 insight)"):
                dist = st.selectbox("District", sorted(nfhs[nfhs["State"] == p["state"]]["District"]))
                need = E.district_needs(nfhs, p["state"], dist, schemes, tindex)
                st.caption("Indicators where this district lags the dataset average the most, with the "
                           "best-matching Central / state schemes for each gap.")
                table(need)

# --------------------------------------------------------------------------- #
# TAB 2 - EDA
# --------------------------------------------------------------------------- #
with tab_eda:
    st.subheader("Scheme dataset")
    e1, e2, e3, e4, e5 = st.columns(5)
    e1.metric("Total schemes", f"{len(schemes):,}")
    e2.metric("Central", int((schemes["level"] == "Central").sum()))
    e3.metric("State / UT", f"{int((schemes['level'] == 'State').sum()):,}")
    e4.metric("Categories", schemes["category_tag"].nunique())
    e5.metric("States / UTs covered", len(STATES))

    a, b = st.columns(2)
    with a:
        cc = schemes["category_tag"].value_counts().reset_index()
        cc.columns = ["Category", "Schemes"]
        show(px.bar(cc, x="Category", y="Schemes", title="Schemes by category", color="Schemes",
                    color_continuous_scale="Greens").update_coloraxes(showscale=False)
             .update_layout(xaxis_tickangle=-40))
    with b:
        sc = schemes["state"].replace({"Not specified": "Unspecified"}).value_counts().reset_index()
        sc.columns = ["State / UT", "Schemes"]
        show(px.bar(sc, x="State / UT", y="Schemes", title="Schemes by state / UT (All India = Central)",
                    color="Schemes", color_continuous_scale="Oranges").update_coloraxes(showscale=False)
             .update_layout(xaxis_tickangle=-60))

    a, b = st.columns(2)
    with a:
        crit = pd.Series({
            "Income limit": schemes["max_annual_income"].notna().mean(),
            "Age limit (min or max)": ((schemes["min_age"] > 0) | (schemes["max_age"] < 120)).mean(),
            "Category-restricted": schemes["caste_list"].apply(bool).mean(),
            "Occupation-tagged": schemes["occ_list"].apply(bool).mean(),
            "Gender-specific": (schemes["gender"] != "All").mean(),
            "Mentions BPL": schemes["bpl_required"].mean(),
        }).sort_values().reset_index()
        crit.columns = ["Criterion", "Share"]
        show(px.bar(crit, x="Share", y="Criterion", orientation="h",
                    title="Data completeness: share of schemes with each structured criterion",
                    color="Share", color_continuous_scale="Greens").update_layout(xaxis_tickformat=".0%")
             .update_coloraxes(showscale=False))
    with b:
        tags = schemes["occ_list"].explode().dropna().value_counts().reset_index()
        tags.columns = ["Beneficiary group", "Schemes"]
        show(px.bar(tags, x="Schemes", y="Beneficiary group", orientation="h", title="Beneficiary groups targeted",
                    color_discrete_sequence=[GREEN]).update_layout(yaxis=dict(autorange="reversed")))

    a, b = st.columns(2)
    with a:
        inc = schemes["max_annual_income"].dropna()
        show(px.histogram(inc, x="max_annual_income", nbins=30, title=f"Income limits (₹) — {len(inc)} schemes",
                          color_discrete_sequence=[SAFFRON]).update_layout(xaxis_title="Annual income limit"))
    with b:
        lim = schemes[(schemes["min_age"] > 0) | (schemes["max_age"] < 120)]
        show(px.histogram(lim[lim["min_age"] > 0], x="min_age", nbins=25, title="Minimum-age requirements",
                          color_discrete_sequence=[NAVY]).update_layout(xaxis_title="Minimum age"))

    st.subheader("NFHS-5 district dataset")
    n1, n2, n3 = st.columns(3)
    n1.metric("Districts", len(nfhs))
    n2.metric("Indicators", len(IND))
    n3.metric("Data points", f"{len(nfhs) * len(IND):,}")
    a, b = st.columns(2)
    with a:
        ind = st.selectbox("Indicator", IND, index=pick("Women literate"))
        show(px.box(nfhs, x="State", y=ind, color="State", points=False, title=f"{ind} (%) by state")
             .update_layout(showlegend=False, xaxis_tickangle=-45))
    with b:
        show(px.imshow(nfhs[IND].corr(), text_auto=".2f", zmin=-1, zmax=1, aspect="auto",
                       color_continuous_scale="RdBu_r", title="Correlation heatmap")
             .update_layout(height=520))
    with st.expander("Browse the raw data"):
        which = st.radio("Dataset", ["Schemes", "NFHS-5 districts"], horizontal=True)
        if which == "Schemes":
            table(schemes[["scheme_name", "level", "state", "category_tag", "min_age", "max_age",
                           "max_annual_income", "gender", "caste_category", "occupation_tags"]])
        else:
            table(nfhs)

# --------------------------------------------------------------------------- #
# TAB 3 - District analytics
# --------------------------------------------------------------------------- #
with tab_dist:
    st.subheader("K-Means district clustering")
    sweep = get_sweep()
    best_k = int(sweep.loc[sweep["Silhouette"].idxmax(), "k"])
    c1, c2 = st.columns([1, 2])
    with c1:
        k = st.slider("Number of clusters (k)", 2, 8, 4)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sweep["k"], y=sweep["Inertia"], name="Inertia", mode="lines+markers",
                                 line_color=NAVY))
        fig.add_trace(go.Scatter(x=sweep["k"], y=sweep["Silhouette"], name="Silhouette", yaxis="y2",
                                 mode="lines+markers", line_color=SAFFRON))
        fig.update_layout(title="Elbow & silhouette", xaxis_title="k", yaxis_title="Inertia",
                          yaxis2=dict(title="Silhouette", overlaying="y", side="right"),
                          legend=dict(orientation="h", y=-0.25))
        show(fig)
        st.caption(f"Highest silhouette in the sweep: k = {best_k}.")
    clustered, profile, sil = get_clusters(k)
    with c2:
        order = sorted(clustered["Cluster"].unique())
        show(px.scatter(clustered, x="PC1", y="PC2", color="Cluster", hover_data=["State", "District"],
                        category_orders={"Cluster": order}, title=f"Districts in PCA space (silhouette = {sil:.2f})",
                        color_discrete_sequence=px.colors.qualitative.Safe))

    prof = profile.drop(columns="Districts")
    z = (prof - prof.mean()) / prof.std(ddof=0).replace(0, 1)
    fig = go.Figure(go.Heatmap(z=z.T.values, x=[f"{i} ({int(profile.loc[i, 'Districts'])})" for i in prof.index],
                               y=prof.columns.tolist(), text=prof.T.round(1).values, texttemplate="%{text}",
                               colorscale="RdYlGn", showscale=False))
    fig.update_layout(title="Cluster profile (mean % per indicator; colour = relative level). Tier 1 = most developed",
                      height=470)
    show(fig)

    st.subheader("What drives an outcome? (Random Forest regression)")
    target = st.selectbox("Outcome to explain", IND, index=pick("Women 10+ yrs schooling"))
    dr = get_drivers(target)
    r1, r2, r3 = st.columns(3)
    r1.metric("R² (test)", f"{dr['r2']:.3f}")
    r2.metric("RMSE (test)", f"{dr['rmse']:.2f} pts")
    r3.metric("5-fold CV R²", f"{dr['cv_r2']:.3f}", f"± {dr['cv_std']:.3f}", delta_color="off")
    a, b = st.columns(2)
    with a:
        imp = dr["importance"].melt(id_vars="Indicator", value_vars=["Importance", "Permutation"], var_name="Method", value_name="Score")
        show(px.bar(imp, x="Score", y="Indicator", color="Method", orientation="h", barmode="group",
                    title=f"Key drivers of '{target}'", color_discrete_sequence=[GREEN, SAFFRON])
             .update_layout(yaxis=dict(autorange="reversed"), legend=dict(orientation="h", y=-0.15)))
    with b:
        lo, hi = float(min(dr["actual"].min(), dr["pred"].min())), float(max(dr["actual"].max(), dr["pred"].max()))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dr["actual"], y=dr["pred"], mode="markers", marker=dict(color=GREEN, opacity=.7),
                                 name="Districts"))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(dash="dash", color="grey"),
                                 name="Perfect fit"))
        fig.update_layout(title="Actual vs predicted (held-out districts)", xaxis_title="Actual", yaxis_title="Predicted")
        show(fig)

    st.subheader("District explorer")
    d1, d2 = st.columns(2)
    sstate = d1.selectbox("State", sorted(nfhs["State"].unique()), key="ex_state")
    sdist = d2.selectbox("District", sorted(nfhs[nfhs["State"] == sstate]["District"]), key="ex_dist")
    row = clustered[(clustered["State"] == sstate) & (clustered["District"] == sdist)].iloc[0]
    st.markdown(f"**{sdist}, {sstate}** belongs to **{row['Cluster']}** (of {k}).")
    table(E.district_needs(nfhs, sstate, sdist, schemes, tindex, top=5))
    sort_col = "Women literate" if "Women literate" in IND else IND[0]
    worst = clustered[clustered["Cluster"] == order[-1]].sort_values(sort_col).head(15)
    with st.expander(f"Priority districts — lowest tier ({order[-1]}), lowest women's literacy first"):
        table(worst[["State", "District"] + IND[:5]])

# --------------------------------------------------------------------------- #
# TAB 4 - Analyze your own data (workbench)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Reading your file…")
def load_user(raw, name, sheet):
    df = A.tidy_columns(A.read_any(raw, name, sheet))
    return A.coerce_numeric_like(df)


@st.cache_data(show_spinner="Clustering…")
def run_cluster(df, cols, k):
    return A.auto_cluster(df, list(cols), k)


@st.cache_data(show_spinner="Running Random Forest regression…")
def run_drivers(df, target, cols):
    return A.drivers(df, target, list(cols))


def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


with tab_work:
    st.subheader("🧪 Analyze your own data")
    st.caption("For officials, NGOs and researchers: upload any district / state / village-level table (NFHS, census, "
               "education, health, survey…) and get a cleaned profile, clusters, key drivers, a ranked priority list "
               "and a scheme-gap view — no coding needed.")
    st.info("🔒 Files are processed in memory for your session only and are not stored by this app. On a public "
            "deployment, do not upload personal or identifiable data — aggregated tables only.")

    up = st.file_uploader("Upload a file (CSV, TSV, TXT, Excel, JSON)", type=["csv", "tsv", "txt", "xlsx", "xls", "json"],
                          key="wb_file")
    use_sample = st.checkbox("…or try it with the bundled NFHS-5 district file", key="wb_sample")
    raw = name = None
    if up is not None:
        raw, name = up.getvalue(), up.name
    elif use_sample:
        raw, name = (E.BASE / "data" / "nfhs_district_wide.csv").read_bytes(), "nfhs_district_wide.csv"

    if raw is None:
        g1, g2, g3 = st.columns(3)
        g1.markdown("**1 · Clean automatically**  \nHandles `*`, `(45.2)`, `1,234`, `%`, duplicates, missing values "
                    "and mixed delimiters.")
        g2.markdown("**2 · Find patterns**  \nCorrelations, group comparison, K-Means clusters and Random Forest "
                    "drivers of any outcome.")
        g3.markdown("**3 · Act on it**  \nRank the areas most in need, spot where few schemes exist, and export "
                    "everything as CSV.")
    else:
        try:
            sheets = A.list_sheets(raw, name)
            sheet = st.selectbox("Sheet", sheets, key="wb_sheet") if len(sheets) > 1 else 0
            df0, converted = load_user(raw, name, sheet)
        except Exception as ex:
            st.error(f"Could not read this file: {ex}")
            st.stop()

        prof0 = A.profile(df0)
        w1, w2, w3, w4 = st.columns(4)
        w1.metric("Rows", f"{prof0['rows']:,}")
        w2.metric("Columns", prof0["cols"])
        w3.metric("Missing cells", f"{prof0['missing_cells']:.1%}")
        w4.metric("Duplicate rows", prof0["duplicates"])
        if converted:
            st.caption("Converted text to numbers in: " + ", ".join(converted[:8]) + (" …" if len(converted) > 8 else ""))

        with st.expander("🧹 Cleaning options", expanded=False):
            o1, o2, o3 = st.columns(3)
            drop_dup = o1.checkbox("Remove duplicate rows", True, key="wb_dup")
            imp_mode = o2.selectbox("Missing numbers", ["median", "mean", "drop", "leave"], key="wb_imp",
                                    format_func=lambda v: {"median": "Fill with median", "mean": "Fill with mean",
                                                           "drop": "Drop those rows", "leave": "Leave as is"}[v])
            max_miss = o3.slider("Drop columns with more missing than", 10, 100, 60, step=5, key="wb_miss") / 100
        df, log = A.clean(df0, drop_dup, imp_mode, max_miss)
        for line in log:
            st.caption("✔ " + line)
        with st.expander("Preview data"):
            table(df.head(50))

        info = A.detect(df)
        non_num = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        s1, s2, s3 = st.columns([3, 1.2, 1.2])
        cols = s1.multiselect("Numeric indicators to analyse", info["numeric"], default=info["numeric"][:15], key="wb_cols")
        label_opts = ["(row number)"] + non_num
        label_col = s2.selectbox("Label column", label_opts, index=label_opts.index(info["name"]) if info["name"] in label_opts else 0,
                                 key="wb_label")
        state_opts = ["(none)"] + non_num
        state_col = s3.selectbox("State column", state_opts,
                                 index=state_opts.index(info["state"]) if info["state"] in state_opts else 0, key="wb_state")
        label_col = None if label_col == "(row number)" else label_col
        state_col = None if state_col == "(none)" else state_col

        if len(cols) < 2 or len(df) < 10:
            st.warning("Select at least 2 numeric indicators and make sure the file has at least 10 rows to run the analyses.")
        else:
            dfa = df.dropna(subset=cols).reset_index(drop=True)
            labels = dfa[label_col].astype(str) if label_col else pd.Series(dfa.index.astype(str))
            (t_ov, t_rel, t_grp, t_cl, t_dr, t_pr, t_gap, t_ex) = st.tabs(
                ["💡 Overview", "🔗 Relationships", "📊 Compare groups", "🧩 Clusters", "🎯 Drivers",
                 "🚩 Priority areas", "🏛️ Scheme gap", "⬇️ Export"])

            with t_ov:
                for line in A.insights(dfa.assign(**{}), cols, label_col, state_col):
                    st.markdown(f"- {line}")
                table(dfa[cols].describe().T.round(2).reset_index().rename(columns={"index": "Indicator"}))
                ind = st.selectbox("Distribution of", cols, key="wb_dist")
                show(px.histogram(dfa, x=ind, nbins=30, marginal="box", title=f"Distribution of {ind}",
                                  color_discrete_sequence=[GREEN]))

            with t_rel:
                sub = cols[:25]
                show(px.imshow(dfa[sub].corr(), text_auto=".2f" if len(sub) <= 14 else False, zmin=-1, zmax=1,
                               aspect="auto", color_continuous_scale="RdBu_r", title="Correlation heatmap")
                     .update_layout(height=560))
                st.markdown("**Strongest relationships**")
                table(A.top_correlations(dfa, cols, 10))
                x1, y1 = st.columns(2)
                xv = x1.selectbox("X axis", cols, key="wb_x")
                yv = y1.selectbox("Y axis", cols, index=min(1, len(cols) - 1), key="wb_y")
                cc = [c for c in non_num if 1 < dfa[c].nunique() <= 12] if non_num else []
                colv = st.selectbox("Colour by (optional)", ["(none)"] + cc, key="wb_col")
                show(px.scatter(dfa.assign(Label=labels), x=xv, y=yv, color=None if colv == "(none)" else colv,
                                hover_name="Label", title=f"{yv} vs {xv}", opacity=.8))

            with t_grp:
                groups = [c for c in non_num if 1 < dfa[c].nunique() <= 60]
                if not groups:
                    st.info("No suitable grouping column (a text column with 2-60 distinct values, e.g. State) found.")
                else:
                    gc, gi = st.columns(2)
                    gcol = gc.selectbox("Group by", groups, index=groups.index(state_col) if state_col in groups else 0,
                                        key="wb_gcol")
                    gind = gi.selectbox("Indicator", cols, key="wb_gind")
                    agg = dfa.groupby(gcol)[gind].agg(["mean", "median", "min", "max", "count"]).round(2) \
                        .sort_values("mean", ascending=False).reset_index()
                    show(px.bar(agg, x=gcol, y="mean", hover_data=["count", "min", "max"], color="mean",
                                color_continuous_scale="Greens", title=f"Average {gind} by {gcol}")
                         .update_coloraxes(showscale=False).update_layout(xaxis_tickangle=-45))
                    show(px.box(dfa, x=gcol, y=gind, points=False, title=f"Spread of {gind} within each {gcol}")
                         .update_layout(xaxis_tickangle=-45))
                    t1, t2 = st.columns(2)
                    rk = dfa.assign(Label=labels)[["Label", gind]].sort_values(gind, ascending=False)
                    t1.markdown(f"**Top 10 — highest {gind}**")
                    with t1:
                        table(rk.head(10))
                    t2.markdown(f"**Bottom 10 — lowest {gind}**")
                    with t2:
                        table(rk.tail(10).iloc[::-1])

            with t_cl:
                cl0 = run_cluster(dfa, tuple(cols), None)
                kk = st.slider("Number of clusters (k)", 2, max(2, min(8, len(dfa) - 1)), cl0["k"], key="wb_k")
                cl = cl0 if kk == cl0["k"] else run_cluster(dfa, tuple(cols), kk)
                cd = cl["data"].copy()
                cd["Label"] = (cd[label_col].astype(str) if label_col else cd.index.astype(str))
                c1, c2 = st.columns([1, 2])
                with c1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=cl["sweep"]["k"], y=cl["sweep"]["Inertia"], name="Inertia",
                                             mode="lines+markers", line_color=NAVY))
                    fig.add_trace(go.Scatter(x=cl["sweep"]["k"], y=cl["sweep"]["Silhouette"], name="Silhouette",
                                             yaxis="y2", mode="lines+markers", line_color=SAFFRON))
                    fig.update_layout(title="Elbow & silhouette", xaxis_title="k",
                                      yaxis2=dict(overlaying="y", side="right"), legend=dict(orientation="h", y=-0.25))
                    show(fig)
                    st.caption(f"Best silhouette at k = {cl['best_k']}. Current silhouette = {cl['silhouette']:.2f}.")
                with c2:
                    show(px.scatter(cd, x="PC1", y="PC2", color="Cluster", hover_name="Label",
                                    category_orders={"Cluster": sorted(cd["Cluster"].unique())},
                                    title="Rows in 2-D (PCA)", color_discrete_sequence=px.colors.qualitative.Safe))
                for c_, d_ in cl["describe"].items():
                    st.markdown(f"- **{c_}** ({int(cl['profile'].loc[c_, 'Rows'])} rows): {d_}")
                z = cl["z"]
                fig = go.Figure(go.Heatmap(z=z.T.values, x=list(z.index), y=list(z.columns),
                                           text=cl["profile"][cols].T.round(1).values, texttemplate="%{text}",
                                           colorscale="RdYlGn", showscale=False))
                fig.update_layout(title="Cluster profile (mean value; colour = relative level)", height=60 + 38 * len(cols))
                show(fig)

            with t_dr:
                if len(dfa) < 30:
                    st.info("At least 30 complete rows are needed for a reliable regression.")
                else:
                    target = st.selectbox("Outcome to explain", cols, key="wb_target")
                    if st.button("Run driver analysis", key="wb_run_dr") or st.session_state.get("wb_dr_done") == target:
                        st.session_state["wb_dr_done"] = target
                        dr = run_drivers(dfa, target, tuple(cols))
                        r1, r2, r3 = st.columns(3)
                        r1.metric("R² (held-out)", f"{dr['r2']:.3f}")
                        r2.metric("RMSE", f"{dr['rmse']:.2f}")
                        r3.metric(f"{dr['folds']}-fold CV R²", f"{dr['cv_r2']:.3f}", f"± {dr['cv_std']:.3f}", delta_color="off")
                        a1, a2 = st.columns(2)
                        with a1:
                            imp = dr["importance"].melt(id_vars="Indicator", value_vars=["Importance", "Permutation"],
                                                        var_name="Method", value_name="Score")
                            show(px.bar(imp, x="Score", y="Indicator", color="Method", orientation="h", barmode="group",
                                        title=f"What drives '{target}'?", color_discrete_sequence=[GREEN, SAFFRON])
                                 .update_layout(yaxis=dict(autorange="reversed"), legend=dict(orientation="h", y=-0.15)))
                        with a2:
                            lo, hi = float(min(dr["actual"].min(), dr["pred"].min())), float(max(dr["actual"].max(), dr["pred"].max()))
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=dr["actual"], y=dr["pred"], mode="markers",
                                                     marker=dict(color=GREEN, opacity=.7), name="Held-out rows"))
                            fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                                     line=dict(dash="dash", color="grey"), name="Perfect fit"))
                            fig.update_layout(title="Actual vs predicted", xaxis_title="Actual", yaxis_title="Predicted")
                            show(fig)
                        st.caption("Importance shows association in this data, not proof of cause and effect.")
                    else:
                        st.caption("Pick an outcome and press **Run driver analysis**.")

            with t_pr:
                st.caption("Builds a 0-100 need score from the indicators you pick (100 = most in need) so you can "
                           "rank districts / areas for targeting.")
                p1, p2 = st.columns(2)
                pcols = p1.multiselect("Indicators in the score", cols, default=cols[:8], key="wb_pcols")
                lower = p2.multiselect("…where LOWER is better (e.g. mortality, poverty)", pcols, key="wb_lower")
                if len(pcols) < 2:
                    st.info("Pick at least two indicators.")
                else:
                    pri = A.priority_index(dfa.assign(Label=labels), pcols, lower)
                    topn = st.slider("Show top", 5, min(100, len(pri)), min(15, len(pri)), key="wb_topn")
                    show_cols = ["Label"] + ([state_col] if state_col else []) + ["Priority score", "Weakest indicators"] + pcols
                    top = pri.head(topn)
                    show(px.bar(top.iloc[::-1], x="Priority score", y="Label", orientation="h", color="Priority score",
                                color_continuous_scale="Reds", title=f"Top {topn} priority areas")
                         .update_coloraxes(showscale=False).update_layout(height=max(350, 24 * topn)))
                    table(top[show_cols])
                    st.download_button("⬇️ Download full ranking (CSV)", csv_bytes(pri[show_cols]), "priority_ranking.csv",
                                       "text/csv", key="wb_dl_pri")

            with t_gap:
                if not state_col:
                    st.info("Choose a State column above to compare your indicator with the number of schemes per state.")
                else:
                    g1_, g2_ = st.columns(2)
                    gm = g1_.selectbox("Indicator", cols, key="wb_gapm")
                    hib = g2_.radio("A higher value means…", ["better outcome", "worse outcome"], horizontal=True,
                                    key="wb_hib") == "better outcome"
                    gap = A.scheme_gap(dfa, state_col, gm, schemes, hib)
                    if gap.empty:
                        st.warning("No state names could be matched. Check the State column.")
                    else:
                        fig = px.scatter(gap, x="State schemes", y="Average", text="State", color="Possible gap",
                                         title=f"{gm}: state average vs number of State-level schemes",
                                         color_discrete_map={"⚠️ high need, fewer schemes": "#c0392b", "high need": SAFFRON,
                                                             "": GREEN})
                        fig.update_traces(textposition="top center", marker=dict(size=12))
                        show(fig)
                        table(gap)
                        st.caption("⚠️ flags states in the worse half on this indicator that also have fewer State-level "
                                   "schemes than the median — candidates for new outreach or scheme design. "
                                   "Counts come from the myScheme dataset in this app; Central schemes apply to all states.")

            with t_ex:
                st.markdown("**Downloads**")
                d1, d2 = st.columns(2)
                d1.download_button("⬇️ Cleaned data (CSV)", csv_bytes(df), "cleaned_data.csv", "text/csv", key="wb_dl_clean")
                clx = run_cluster(dfa, tuple(cols), None)["data"]
                d2.download_button("⬇️ Data with cluster labels (CSV)", csv_bytes(clx.drop(columns=["PC1", "PC2"])),
                                   "clustered_data.csv", "text/csv", key="wb_dl_cl")
                st.markdown("**What was done to your file**")
                st.markdown("\n".join(f"- {x}" for x in (log or ["No cleaning changes were needed."]) +
                                      ([f"Converted text columns to numbers: {', '.join(converted)}"] if converted else [])))

# --------------------------------------------------------------------------- #
# TAB 5 - Model evaluation
# --------------------------------------------------------------------------- #
with tab_model:
    st.subheader("Eligibility classifier: Logistic Regression vs Random Forest")
    ok = rf_res["Accuracy"] >= 0.90
    (st.success if ok else st.warning)(
        f"Random Forest test accuracy: **{rf_res['Accuracy']:.2%}**  "
        f"({'meets' if ok else 'below'} the 90% target) — measured live on "
        f"{bundle['n_test']:,} unseen citizen–scheme pairs.")
    m = st.columns(5)
    for col, key in zip(m, ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]):
        col.metric(key, f"{rf_res[key]:.3f}" if key == "ROC-AUC" else f"{rf_res[key]:.1%}",
                   f"{rf_res[key] - lr_res[key]:+.3f} vs LR", delta_color="off")

    comp = pd.DataFrame([{k: v for k, v in r.items() if k in ("Model", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC")}
                         for r in bundle["results"]])
    a, b = st.columns(2)
    with a:
        show(px.bar(comp.melt("Model", var_name="Metric", value_name="Score"), x="Metric", y="Score", color="Model",
                    barmode="group", title="Model comparison", range_y=[0, 1.05],
                    color_discrete_sequence=[SAFFRON, GREEN]))
    with b:
        fig = go.Figure()
        for r, c in zip(bundle["results"], [SAFFRON, GREEN]):
            fig.add_trace(go.Scatter(x=r["fpr"], y=r["tpr"], mode="lines", name=f"{r['Model']} (AUC {r['ROC-AUC']:.3f})",
                                     line=dict(color=c, width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="grey"), name="Chance"))
        fig.update_layout(title="ROC curves", xaxis_title="False-positive rate", yaxis_title="True-positive rate",
                          legend=dict(orientation="h", y=-0.25))
        show(fig)

    a, b = st.columns(2)
    for col, r in zip((a, b), bundle["results"]):
        with col:
            show(px.imshow(r["cm"], text_auto=True, x=["Not eligible", "Eligible"], y=["Not eligible", "Eligible"],
                           labels=dict(x="Predicted", y="Actual"), color_continuous_scale="Greens",
                           title=f"Confusion matrix — {r['Model']}").update_coloraxes(showscale=False))

    a, b = st.columns(2)
    with a:
        sg = bundle["segments"]
        sg = sg[sg["Group"] == "Category"].sort_values("Accuracy")
        show(px.bar(sg, x="Accuracy", y="Segment", orientation="h", hover_data=["Test pairs"],
                    title="Random Forest accuracy by scheme category", range_x=[0.8, 1.0],
                    color="Accuracy", color_continuous_scale="Greens")
             .update_layout(height=640).update_coloraxes(showscale=False))
    with b:
        imp = bundle["importance"].sort_values().reset_index()
        imp.columns = ["Feature", "Importance"]
        show(px.bar(imp, x="Importance", y="Feature", orientation="h", title="Top features used by the Random Forest",
                    color_discrete_sequence=[GREEN]).update_layout(height=640))

    with st.expander("📝 How this accuracy is measured (read before presenting)"):
        st.markdown(f"""
- **Task:** given a citizen's profile and a scheme's attributes, predict *Eligible / Not eligible*.
- **Data:** {bundle['n_profiles']:,} simulated citizens, each paired with random schemes from the {len(schemes):,}-scheme
  dataset. Only {bundle['raw_pos_rate']:.0%} of random pairs are eligible, so the set was **balanced to 50 / 50**
  ({bundle['n_pairs']:,} pairs) — a model that always says "not eligible" would score only 50%, not 95%.
- **Labels** come from the rule-based eligibility engine (age, gender, income, category, occupation, state, BPL).
- **No leakage:** the train/test split is by *citizen*, so the models are tested on people they never saw.
- **Raw features:** the profile and the scheme's limits are given as inputs. The only pre-computed comparison is
  `state_ok` (does the citizen's state equal the scheme's state?).
- **What it proves:** the ML model learned the eligibility logic and generalises to new profiles. It does not measure how
  well the structured fields reflect the official scheme text (only ~7-10% of schemes have an age or income limit
  recorded), which is why the app also shows the official eligibility text.
- Logistic Regression is linear, so it struggles with interacting conditions; the Random Forest captures them and
  supplies the *AI confidence* shown next to each result.
""")

# --------------------------------------------------------------------------- #
# TAB 6 - About
# --------------------------------------------------------------------------- #
with tab_about:
    st.subheader("Problem")
    st.write("Scheme information is scattered across many portals and the eligibility rules are buried in long text, "
             "so citizens often do not know which schemes apply to them and benefits go unclaimed.")
    st.subheader("Solution — 7-stage pipeline")
    st.markdown("""
1. **User input** — age, gender, state/UT, category, income, occupation and situation flags
2. **Data preprocessing** — de-duplication, standardised labels, invalid ages/incomes removed, eligibility conditions turned into structured attributes
3. **Exploratory data analysis** — distributions, correlation heat-map, rule prevalence
4. **Eligibility filtering** — rule engine removes schemes whose conditions are not met
5. **Profile matching & ranking** — match score from targeting specificity, interests and free-text relevance
6. **Machine-learning analytics** — Logistic Regression / Random Forest (eligibility), K-Means + Random Forest regression (NFHS-5)
7. **Output & deployment** — Streamlit app, GitHub, Streamlit Community Cloud

**Analyze Your Data:** a workbench where officials and researchers can upload their own table (CSV, Excel, JSON…)
and get automatic cleaning, clustering, driver analysis, a priority ranking and a scheme-gap view.
""")
    st.subheader("Match score")
    st.write("Eligible schemes start at 50. Up to +27 is added for how *targeted* the scheme is at you (age window, "
             "category, occupation, income cap, BPL, your own state), +10 if it is in your areas of interest, and up to "
             "+12 for TF-IDF text similarity with your description. Maximum 99.")
    st.subheader("Scaling up")
    st.write("Drop an updated scheme CSV with the same columns into `data/schemes_processed.csv` (or an NFHS file with the same "
             "layout into `data/nfhs_district_wide.csv`) and every chart, model and metric recomputes automatically.")
    st.caption("Developed by Dharanika V (24DSC238) · B.Sc. Data Science · The American College, Madurai")
