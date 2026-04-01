test_data = [
    # ── Attention Paper — Factual ──────────────────────────────
    {
        "question": "What architecture does the Transformer use instead of recurrence and convolutions?",
        "ground_truth": "The Transformer relies entirely on attention mechanisms to draw global dependencies between input and output, dispensing with recurrence and convolutions entirely."
    },
    {
        "question": "How many encoder layers does the Transformer base model have?",
        "ground_truth": "The encoder is composed of a stack of N=6 identical layers."
    },
    {
        "question": "What optimizer was used to train the Transformer and what were its parameters?",
        "ground_truth": "Adam optimizer with β1=0.9, β2=0.98 and ε=10^-9 with warmup_steps=4000."
    },
    {
        "question": "Why did the authors use sinusoidal positional encoding instead of learned embeddings?",
        "ground_truth": "They chose sinusoidal positional encoding because it may allow the model to extrapolate to sequence lengths longer than those encountered during training."
    },
    {
        "question": "How long did it take to train the Transformer big model?",
        "ground_truth": "The big model was trained for 300,000 steps which took 3.5 days on 8 NVIDIA P100 GPUs."
    },

    # ── Attention Paper — Table based ──────────────────────────
    {
        "question": "What is the BLEU score of the Transformer big model on English to German translation?",
        "ground_truth": "28.4 BLEU on the WMT 2014 English-to-German translation task."
    },
    {
        "question": "What is the complexity per layer of Self-Attention compared to Recurrent layers?",
        "ground_truth": "Self-Attention has complexity O(n²·d) per layer while Recurrent layers have O(n·d²) per layer."
    },
    {
        "question": "What BLEU score did the Transformer base model achieve on English to French translation?",
        "ground_truth": "38.1 BLEU on English to French translation."
    },

    # ── CN PPTX ────────────────────────────────────────────────
    {
        "question": "What are the course objectives of the Computer Networks course?",
        "ground_truth": "To introduce fundamental concepts of data communications and computer networks, describe layers protocols and services in ISO-OSI and TCP/IP models, and enable students to use client/server architecture and implement network routing algorithms."
    },
    {
        "question": "What protocols are students expected to illustrate as part of the CN course outcomes?",
        "ground_truth": "Students are expected to illustrate different protocols including DNS, HTTP, SMTP, SNMP and FTP."
    },
]