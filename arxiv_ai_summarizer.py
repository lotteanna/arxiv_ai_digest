#!/usr/bin/env python3

# VERSION 1.0 - PREFERRED
# Description: Focused AI paper search, summarization, and ranking.
# Features: 
# - Searches for 200 recent AI papers
# - Summarizes papers (100-200 words)
# - Ranks based on relevance to specific AI concepts
# - Selects top 15 papers for digest
# - Emails digest to specified address

import arxiv
from datetime import datetime, timezone
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv
import torch
import numpy as np
import time

# Initialize the models
device = 0 if torch.cuda.is_available() else -1
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load environment variables
load_dotenv()

def search_arxiv():
    base_query = ('(cat:cs.AI OR cat:cs.CL OR cat:cs.CV OR cat:cs.LG OR cat:cs.MM OR cat:stat.ML) AND '
                  '("generative AI" OR "large language models" OR "multimodal AI" OR '
                  '"graph neural networks" OR "AI applications" OR '
                  '"natural language processing" OR "reinforcement learning" OR '
                  '"machine learning" OR "artificial intelligence")')
    
    client = arxiv.Client()
    search = arxiv.Search(
        query=base_query,
        max_results=200,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    results = list(client.results(search))
    print(f"Total papers found: {len(results)}")
    return results

def summarize_paper(paper):
    summary = summarizer(paper.summary, max_length=200, min_length=100, do_sample=False)[0]['summary_text']
    first_author = paper.authors[0].name if paper.authors else "Unknown"
    return f"Title: {paper.title}\nFirst Author: {first_author}\nSummary: {summary}\nURL: {paper.entry_id}\n\n"

def rank_papers(papers):
    target_concepts = [
        "groundbreaking advances in generative AI",
        "innovative applications of AI in business",
        "novel AI techniques for recommendation systems",
        "advancements in natural language processing",
        "cutting-edge AI for content creation",
        "AI-driven business intelligence",
        "breakthroughs in multimodal AI systems",
        "revolutionary AI architectures",
        "AI applications in voice synthesis",
        "graph-based AI techniques"
    ]
    target_embeddings = sentence_model.encode(target_concepts, convert_to_tensor=True)
    
    scored_papers = []
    for paper in papers:
        text_to_check = paper.title + ' ' + paper.summary
        paper_embedding = sentence_model.encode(text_to_check, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(paper_embedding, target_embeddings).max().item()
        scored_papers.append((paper, similarity))
    
    return sorted(scored_papers, key=lambda x: x[1], reverse=True)

def send_email(content):
    sender_email = os.getenv('SENDER_EMAIL')
    receiver_email = os.getenv('RECEIVER_EMAIL')
    password = os.getenv('EMAIL_PASSWORD')

    if not all([sender_email, receiver_email, password]):
        print("Email credentials not set. Please check your .env file.")
        return

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = f"AI Research Digest: Groundbreaking AI Research and Applications - {datetime.now().strftime('%Y-%m-%d')}"

    message.attach(MIMEText(content, "plain"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, password)
            server.send_message(message)
        print("Email sent successfully")
    except Exception as e:
        print(f"Failed to send email: {str(e)}")

def generate_digest():
    print("Preparing AI research digest...")
    papers = search_arxiv()
    
    if not papers:
        print("No relevant papers found.")
        return

    ranked_papers = rank_papers(papers)
    top_papers = ranked_papers[:15]  # Get top 15 papers for the digest
    
    summaries = [f"Relevance Score: {score:.2f}\n{summarize_paper(paper)}" for paper, score in top_papers]
    
    if summaries:
        content = "Most Relevant AI Research and Applications:\n\n" + "\n".join(summaries)
        send_email(content)
    else:
        print("No relevant innovative AI papers found.")
    
    print("AI research digest completed.")

if __name__ == "__main__":
    print("Starting the ArXiv AI research digest...")
    generate_digest()
    print("Digest process completed.")