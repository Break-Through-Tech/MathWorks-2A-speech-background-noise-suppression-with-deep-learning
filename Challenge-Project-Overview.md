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
**Challenge Advisor:** Neha Sardesai, nsardesa@mathworks.com.   
**AI Studio Coach:** Bhavya Gopal, bhavya.gopal@breakthroughtech.org   
**Program:** Break Through Tech AI Studio - Fall 2026  

---

## 🏢 About MathWorks
MathWorks is the leading developer of mathematical computing software for engineers and scientists worldwide. Their flagship products, MATLAB and Simulink, empower innovators to accelerate the pace of engineering and scientific discovery across industries such as automotive, aerospace, and communications. The team objectives for this challenge focus on enhancing speech processing capabilities to improve real-world accessibility and communication clarity.

---

## 🎯 The Challenge
### Project Summary
In this project, you will use labeled speech and background noise audio data (from the Microsoft DNS Challenge dataset) and deep learning techniques (CNNs or RNNs) with signal processing and feature extraction to build and validate a model that suppresses background noise from speech signals in real time. This will help MathWorks advance speech enhancement tools that improve the quality of life for people with hearing impairments and noise suppression in online meeting environments.

### Success Criteria

_Objective Metrics:_
- PESQ (Perceptual Evaluation of Speech Quality) — standard ITU metric for quantifying speech quality improvement after denoising
- STOI (Short-Time Objective Intelligibility) — measures how intelligible the cleaned speech is

A successful outcome would show meaningful improvement in both scores compared to the noisy (unprocessed) baseline.

_Subjective Evaluation:_
- Informal listening tests where the denoised audio is compared against the noisy input — the cleaned speech should sound noticeably clearer with reduced background interference.

_A successful December outcome looks like:_   
- A trained MATLAB-based deep learning model that (1) demonstrably reduces background noise across varied noise types from the DNS dataset, (2) achieves competitive PESQ/STOI scores relative to published benchmarks, and (3) is documented well enough that the approach and results can be clearly presented — ideally with a live or recorded demo.

### Stretch Goals

1. Deploy the trained model to process audio in real time using streaming audio capabilities — this is the most impactful stretch goal given the hearing aid application context.
2. Evaluate robustness on noise types outside the DNS dataset — cafeteria noise, music, wind — to test real-world applicability.

### Project Milestones
Use these milestones to guide your work. Your team will create a GitHub Projects board to track tasks within each milestone.

| Month | Milestone | Key Activities |
|---|---|---|
| September | Foundation & Data Pipeline | Get the environment set up and data ready. Students should complete a literature review on deep learning-based noise suppression, download and explore the Microsoft DNS Challenge dataset, and implement basic audio preprocessing and feature extraction (e.g., spectrograms). |
| October | Model Design & Training | Design and train the deep learning network. Students should select an architecture (CNN or RNN), train on the processed dataset, and iterate on hyperparameters. By end of month, the model should produce preliminary denoised audio outputs. |
| November | Evaluation & Delivery | Validate the trained model using both subjective (listening tests) and objective metrics (e.g., PESQ, STOI). Students should document results, reflect on limitations, and prepare a final demo or presentation showcasing the model's performance on real-world noisy speech samples. |

> **Note for the team:** Please create a GitHub Projects board in this repository to break these milestones into weekly tasks. Go to the **Projects** tab → **New project** → Choose **Board** → Add columns for each month.

---

## 📊 Dataset
**Name and Source:** Microsoft DNS Challenge (Deep Noise Suppression)   
**Format:** Audio files (e.g, .mp3, .wav)   
**Size:** over 10gb  
**Location:** https://github.com/microsoft/DNS-Challenge

### Key Details
- [Brief description of what's in the data]
- [Any known limitations or preprocessing needed]
- [Link to data dictionary or documentation, if available]
  
---

## 🛠️ Suggested Approach

**ML Problem Type:** Regression, Time Series Analysis, Deep Learning / Neural Networks, Transfer Learning / Pre-trained Models

**Recommended Libraries:**
- [e.g., pandas, scikit-learn, TensorFlow, Hugging Face]

**Evaluation Metrics:**
- [e.g., Accuracy, Precision/Recall, RMSE, BLEU score]
  
---

## 📚 Resources to Get Started

The following resources will help your team understand the problem space and potential technical approaches for this project:

**Background Reading:**
- [e.g., Link to an article or blog post about the problem domain]
- [e.g., Link to an industry report or case study]

**Technical Tutorials:**
- [e.g., Link to a free tutorial on the ML technique(s) involved]
- [e.g., Link to documentation for a key library or tool]

**Code Examples:**
- [e.g., Link to a relevant GitHub repo]
- [e.g., Link to a sample implementation or starter code]

**Other:**
- [Links to any additional resources — e.g., papers, videos, podcasts, etc.]

*Feel free to explore beyond these, and share anything interesting you find with me!*

---

## 🤝 How We'll Work Together

**Official check-ins:** During our biweekly 45-minute AI Studio Lab Section meeting block (2nd and 4th week of every month)

 **Other ways to reach out to me with questions:** 
* [e.g., Your team's channel within Break Through Tech’s Discord space]
* [e.g., Email; please copy your teammates and AI Studio Coach]
* [e.g., Request a team check-in on Zoom]
* [Note: I will aim to respond within 48 hours. Please reach out to your AI Studio Coach with urgent questions.]

> 💡 **Challenge Advisor: Please update the above based on your availability and preference. If you are not able to answer questions or meet with fellows outside of the biweekly Lab Section check-ins, simply write in "N/A (only available during the official check-in times)"**

**Recommended free coding / collaboration tools**
* […]
* […]

---

## 🚀 Getting Started

1. **Review this overview document** and note any questions for our first meeting
2. **Begin reviewing the dataset** using the link above
3. **Read the GitHub Projects documentation** [here](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects)

I’m excited to work with you!

---

## ❓ Questions?

Please bring any questions to our first meeting during the week of August 24th (Break Through Tech’s Bridge to Studio - Session C). 
