# SBI Hackathon @ Grenoble 2026

Welcome to the **Simulation-Based Inference (SBI) Hackathon in Grenoble** — a three-day, hands-on event dedicated to exploring modern Bayesian inference tools for complex scientific simulators.

**📅 Dates:** 21-23 January 2026  

**📍 Location:** IMAG building, Université Grenoble Alpes campus, Grenoble, France

**🧩 Duration:** 3 days (1.5 days tutorial + 1.5 days hackathon)  

**👥 Target audience:** researchers, students, and practitioners interested in parameter estimation, model calibration, or uncertainty quantification in physical, biological, or engineering simulations.

**Registration:** Free, but limited spots. Please register [here](https://forms.gle/iYQVGuWKisfnwaWn8) by 31 December 2025. We will come back to you to confirm your registration.

---

## 🔍 Motivation

Many modern scientific problems rely on complex, stochastic simulators for which classical likelihood-based inference is intractable. **Simulation-Based Inference (SBI)** provides a principled and efficient Bayesian framework to learn from such models — enabling posterior estimation and parameter calibration directly from simulated data.

This hackathon aims to lower the barrier for new users by combining a **guided tutorial** and an **open hack session** where participants apply SBI to their own problems.

## 📚 Structure

### **Day 1–2 (Tutorial, 1.5 days)**
Led by **[Jan Teusen](https://github.com/janfb)**. Topics include:
- Generative modeling and the SBI framework  
- Neural posterior estimation (NPE), likelihood and ratio estimation  
- Hands-on notebooks using the [`sbi`](https://github.com/mackelab/sbi) Python package  
- Applying SBI to toy and benchmark simulators

### **Day 2–3 (Hackathon, 1.5 days)**
Participants form small teams around real or toy simulators (your own, or provided examples).  
Possible directions:
- Adapting SBI to your research simulator  
- Benchmarking different inference strategies  
- Building reproducible pipelines for calibration or uncertainty quantification  
- Sharing and discussing results at the closing session

We will take inspiration from successful open hackathons such as the [IGE-Jaxathon 2025](https://github.com/Diff4Earth/ige-jaxathon-2025), emphasizing collaboration, openness, and applied research.

## (tentative) Schedule

**-- Wednesday 21-Janvier 2026**
08h30-09h00 : Welcome
09h00-10h30 : <span style="color:green; font-weight:bold">(Tutorial)</span> Session 1
10h30-10h45 : Coffee break
10h45-12h00 : <span style="color:green; font-weight:bold">(Tutorial)</span> Session 2
12h00-13h30 : Lunch
13h30-15h00 : <span style="color:green; font-weight:bold">(Tutorial)</span> Session 3
15h00-15h30 : Goûter
15h30-17h00 : <span style="color:green; font-weight:bold">(Tutorial)</span> Session 4

**-- Thursday  22-Janvier 2026**
08h30-09h00 : Welcome
09h00-10h30 : <span style="color:green; font-weight:bold">(Tutorial)</span> Session 5
10h30-10h45 : Coffee break
10h45-12h00 : <span style="color:green; font-weight:bold">(Tutorial)</span> Session 6
12h00-13h30 : Lunch
13h30-15h00 : <span style="color:blue; font-weight:bold">(Hackathon)</span>
15h00-15h30 : Goûter
15h30-17h00 : <span style="color:blue; font-weight:bold">(Hackathon)</span>

**-- Friday  23-Janvier 2026**
08h30-09h00 : Welcome
09h00-10h30 : <span style="color:blue; font-weight:bold">(Hackathon)</span>
10h30-10h45 : Coffee break
10h45-12h00 : <span style="color:blue; font-weight:bold">(Hackathon)</span>
12h00-13h30 : Lunch
13h30-15h00 : <span style="color:blue; font-weight:bold">(Hackathon)</span>
15h00-15h30 : Goûter
15h30-17h00 : <span style="color:blue; font-weight:bold">(Hackathon)</span>

---

## ⚙️ Practical Information

- **Prerequisites:** basic Python and machine learning familiarity
- **Environment:** see Installation section below
- **Code & materials:** shared through this repository and GitHub Classroom
- **Outputs:** participants are encouraged to share notebooks, results, and ideas for follow-up collaborations.

---

## 🚀 Installation

### Recommended: Using `uv`

We recommend using [**uv**](https://github.com/astral-sh/uv) — a fast, modern Python
package manager. Install it following the instructions on their site, then run:

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

### Alternative: Using conda or mamba

If you prefer conda/mamba, create an environment and install via pip:

```bash
conda create -n sbi-hackathon python=3.10
conda activate sbi-hackathon
pip install -e .
```

Or use `mamba` as a drop-in replacement for faster installation.

---

## 💬 Contact & Updates

Organized by the [SBI4C chair](https://sbi4c.inria.fr/) from the MIAI institute.

Stay tuned and get ready to bring your simulator, your curiosity, and your questions!

---

> “From toy models to real-world calibration — discover how SBI can unlock your simulations.”
