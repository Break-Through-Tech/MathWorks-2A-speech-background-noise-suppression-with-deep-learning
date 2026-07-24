---

> ## Challenge Advisor: Update & Finalize Your Project Overview
>
> > 💡 **These grey text instructions are just for you, the team's Challenge Advisor; please delete them once you have completed the steps below.**
>
> We've pre-populated this Challenge Project Overview page — which is what will be shared with your Break Through Tech student team in August — using the details from your submission form. You should have received an email inviting you to join this repo as a Collaborator, enabling you to add files and make edits.
> 
> In order for your project to be finalized and assigned to a team, please:
> 1. **Review all sections below** and update or expand any content as needed, making sure to address the SME Feedback in the section immediately below. Look for square brackets to find the places below that require additional inputs from you (e.g., "About [Company / Org Name]").
> 2. **Add your dataset** to the [data folder](data) in this repo.
> 3. **Close the Issue assigned to you in this repo** to let us know that you have made your edits and the overview page is ready for final review. You can do this by going to the _Issues_ tab in the top left section of the menu above, add a comment that says "CA review complete", and click the button to Close the Issue. 
>
> If you're unfamiliar with how to edit a page like this in GitHub, check out [this tutorial](https://ubc-lib-geo.github.io/gis-workshop-waml-template/content/handson/edit-readme.html) for a quick overview (start with step 2 and only edit this page), and [this guide](https://ubc-lib-geo.github.io/gis-workshop-waml-template/content/markdown.html) on how to use Markdown to compose text.
>
>
> ❌ Remember that this is a public repo. Do NOT include: Proprietary data, PII, API keys, credentials, or anything confidential.

---

## 📋 BTT Internal Evaluation Notes
*(This section is for BTT staff and CAs only — remove before sharing with students)*

### Technical Vetting
| Check | Status | Notes |
| :--- | :--- | :--- |
| Python Compatibility | 🟡 | The project explicitly states MATLAB as the tech stack for the model, which is not compatible with the Python-centric BTT ecosystem. While signal processing and deep learning concepts can be implemented in Python, the deliverable artifact is tied to MATLAB. |
| Data Readiness | 🟡 | The Microsoft DNS Challenge dataset is generally well-curated. However, audio data processing and feature extraction (e.g., spectrograms, MFCCs) can be complex and time-consuming, potentially consuming significant student effort. |
| Resource Check | 🔴 | The project requires specialized compute resources (GPU/TPU) and explicitly mentions MATLAB, which is proprietary software. This violates the free-tier Colab and no proprietary software constraints. |

### Internal Scores
- **Student Fit Score:** 5/10
- **Technical Depth Score:** 8/10
- **Overall Recommendation:** REVISE

### Advisor Feedback Draft
Thank you for submitting the Speech Background Noise Suppression project proposal. The application of deep learning for audio enhancement is a technically sophisticated area, and the use of objective metrics like PESQ and STOI demonstrates a commitment to rigorous evaluation. However, to align with BTT's core principles and ensure a high-signal learning experience for fellows, we need to address a few key areas.

Firstly, the reliance on MATLAB as the primary tech stack for the model deliverable poses a significant challenge, as BTT's curriculum is entirely Python-based. A potential adjustment would be to scope the project such that the *research* and *prototyping* utilize Python libraries (e.g., TensorFlow/Keras, PyTorch, Librosa) for all modeling and evaluation, with the final MATLAB integration being a separate, potentially out-of-scope, task for the partner.

Secondly, the explicit requirement for GPU/TPU resources and the mention of proprietary software (MATLAB) currently fall outside our free-tier Google Colab and open-source constraints. We recommend pivoting to techniques that are feasible within these limitations. For example, exploring smaller, more efficient model architectures or focusing on sophisticated signal processing techniques that can be implemented and trained within standard Colab environments.

Thirdly, while the Microsoft DNS dataset is valuable, the complexity of audio preprocessing and feature engineering for speech enhancement can be substantial. To ensure a 'Goldilocks' difficulty, we might need to narrow the scope of noise types or focus on a specific, well-defined audio processing pipeline rather than a broad 'suppresses background noise' goal.

We are eager to explore how this project can be adapted to provide a rich, Python-centric deep learning experience for our fellows. Please let us know your thoughts on these adjustments, and we can schedule a follow-up to discuss the path forward.

---

# Speech Background Noise Suppression with Deep Learning

**Company / Org:** MathWorks  
**Challenge Advisor:** Neha Sardesai, nsardesa@mathworks.com  
**Program:** Break Through Tech AI Studio - Fall 2026  

---

## 🏢 About MathWorks
MathWorks is the leading developer of mathematical computing software for engineers and scientists worldwide. Their flagship products, MATLAB and Simulink, empower innovators to accelerate the pace of engineering and scientific discovery across industries such as automotive, aerospace, and communications. The team objectives for this challenge focus on enhancing speech processing capabilities to improve real-world accessibility and communication clarity.

---

## 🎯 The Challenge
### Project Summary
This project leverages the Microsoft DNS Challenge dataset to build and validate a deep learning model capable of real-time background noise suppression. By integrating neural architectures like CNNs or RNNs with traditional signal processing techniques, the team will develop a system that significantly improves speech quality. The final outcome aims to deliver a robust solution that assists in creating clearer audio environments for individuals with hearing impairments and professionals in high-noise workspaces.

### Success Criteria
Objective: PESQ (Perceptual Evaluation of Speech Quality) and STOI (Short-Time Objective Intelligibility). Subjective: Informal listening tests compared against the noisy input.

### Project Milestones
Use these milestones to guide your work. Your team will create a GitHub Projects board to track tasks within each milestone.
| Month | Milestone | Key Activities |
|-------|-----------|----------------|
| **September** | Data Exploration & Preprocessing | Load audio datasets, establish a pipeline for waveform visualization, and identify noise/speech separation protocols. |
| **October** | Feature Engineering & Baseline Modeling | Implement audio feature extraction (spectrograms/MFCCs) and train an initial baseline CNN or RNN architecture. |
| **November** | Model Optimization & Evaluation | Execute iterative hyperparameter tuning, conduct validation runs using PESQ/STOI, and refine model architecture. |
| **December** | Insights, Deliverables & Presentation | Finalize model documentation, package code, and prepare the end-of-program presentation and demo. |

> **Note for the team:** Please create a GitHub Projects board in this repository to break these milestones into weekly tasks. Go to the **Projects** tab → **New project** → Choose **Board** → Add columns for each month.

---

## 📊 Dataset
**Name and Source:** Microsoft DNS Challenge (Deep Noise Suppression)  
**Format:** Raw Audio (.wav) and Metadata  
**Size:** over 10gb  
**Location:** Internal partner link provided via project kickoff documentation.  

### Key Details
- Microsoft DNS Challenge dataset - labeled speech and background noise audio data.
- The data requires careful handling of sample rates, windowing for Fourier Transforms, and normalization to ensure consistency during the training of the neural network.

---

## 🛠️ Suggested Approach
**ML Problem Type:** Regression / Signal Processing  
**Recommended Libraries:**
- Deep Learning (CNNs, RNNs)
- Signal Processing Toolkits
- MATLAB
**Evaluation Metrics:** PESQ (Perceptual Evaluation of Speech Quality), STOI (Short-Time Objective Intelligibility)

---

## 📚 Resources to Get Started
The following resources will help your team understand the problem space and potential technical approaches for this project:
**Background Reading:**
- Documentation on the Microsoft DNS Challenge benchmarks and speech enhancement architectures.
**Technical Tutorials:**
- MathWorks documentation on Deep Learning for audio and signal processing.
**Code Examples:**
- Sample starter notebooks for audio preprocessing and basic CNN implementation.

---

## 🤝 How We'll Work Together
**Check-ins:** During our biweekly 60-min AI Studio Lab Section meeting block (2nd and 4th week of every month)  
**Communication:** Email and scheduled platform-specific syncs.  
**Response time:** 24–48 business hours.  
**Recommended Tools:**
- **Coding:** Google Colab Free Tier  
- **Collaboration:** GitHub, Notion  
- **Virtual Meetings:** Zoom, Google Meet  

---

## 🚀 Getting Started
1. **Review this overview document** and note any questions for our first meeting.
2. **Begin reviewing the dataset** using the link provided in the Dataset section.
3. **Read the GitHub Projects documentation** [here](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects).

I'm excited to work with you!

---

## ❓ Questions?
Please bring any questions to our first meeting during the week of August 24th (Break Through Tech's Bridge to Studio - Session B).
