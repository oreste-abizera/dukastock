# DukaStock — Privacy Policy and End-User Licence Agreement (EULA)

**Effective for:** the DukaStock WhatsApp and USSD sales-logging service
**Data controller:** Oreste Abizera, BSc. Software Engineering capstone researcher, African Leadership University
**Contact:** oresteabizera11@gmail.com
**Supervisor:** Hubert Apana (hapana@alueducation.com)

This page exists as the dedicated Privacy Policy / EULA artefact required for this project's ethics review and defense. It is written in plain language on purpose, since the shopkeepers who use this system communicate in informal, conversational Kinyarwanda and English, not legal English.

---

## 1. What data DukaStock collects

- **Message content** you send over WhatsApp or USSD to log a sale (for example, "nagurishije isukari kilo 2").
- **Your phone number**, which is hashed immediately on arrival and never stored or viewed in its original, readable form.
- **Sales records** derived from your messages: product, quantity, unit, and timestamp.

DukaStock does not collect your name, location, national ID, or any payment or financial account information. It does not request access to your contacts, photos, or anything else on your phone.

## 2. Why this data is collected (purpose of processing)

Data is collected for two purposes only: (1) to operate the sales-logging service you are actively using (recording and retrieving your own sales history, and giving restocking advice), and (2) for the research this system was built for — evaluating whether a language model can correctly understand informal commerce messages. Research use is limited to the annotated message corpus already reviewed and approved by the ALU Research Ethics Committee; it does not extend to using your live production messages for new, unreviewed research without further ethical review.

## 3. Who your data is shared with

Your messages pass through two third-party communication providers strictly to deliver the service: **Twilio** (for WhatsApp) and **Africa's Talking** (for USSD and SMS). These providers see the message in transit as part of routing it, under their own standard telecom/API terms of service. DukaStock does not sell, rent, or otherwise share your data with advertisers, data brokers, or any other third party.

## 4. How your data is protected

Your phone number is hashed at the moment it is received, under Rwanda's Law No. 058/2021 on the Protection of Personal Data and Privacy, and is never stored in plaintext anywhere in the system. Message text used for research annotation is handled entirely on a self-hosted, local annotation tool — it is never uploaded to a third-party annotation or labelling service.

## 5. How long data is kept

Sales records are kept for as long as the service is operated, so that you can retrieve your own sales history. Research-annotated message data is retained only for the duration of this capstone project and any explicitly REC-approved follow-on research; it is not kept indefinitely or repurposed without further review.

## 6. Your rights

Under Rwanda's Law No. 058/2021, you have the right to be informed about how your data is processed, to ask what data is held about you, and to request correction or deletion. Because your phone number is hashed, deletion requests should be made by contacting the data controller directly (above) with enough detail (e.g., approximate dates and shop details) to locate your records, since the system itself cannot reverse a hash to look you up by phone number.

## 7. Consent and withdrawal

By sending a message to the DukaStock WhatsApp number or dialling its USSD code, you are using the service voluntarily. If any of your messages were part of the annotated research dataset, you were told at the time of collection that your message would be used for this research, and you may withdraw that research consent at any time by contacting the data controller — this does not affect your ability to keep using the live sales-logging service itself.

## 8. Model accuracy and limitation of liability

DukaStock uses a machine-learning model (fine-tuned XLM-R) to interpret your message, with a rule-based fallback (RapidFuzz) when the model is not confident. **This system can make mistakes** — it may occasionally record the wrong product, quantity, or unit, particularly for products, phrasing, or dialects it has not seen in its training data. DukaStock is a research prototype, not a certified point-of-sale or accounting system, and it should not be treated as your sole or authoritative business record without spot-checking. The developer is not liable for business decisions made solely on the basis of a misread entry; you are encouraged to verify important totals yourself.

## 9. Changes to this policy

Any material change to what data is collected or how it is used will be reflected in an updated version of this document before that change takes effect.

---

*This document satisfies the mandatory EULA/Privacy Policy artefact requirement for the Ethics in Software Engineering summative assessment. It must be opened and its major clauses walked through and explained on camera during the required video presentation — a written policy alone does not satisfy the assessment's scoring rule.*
