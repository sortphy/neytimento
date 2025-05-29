#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para baixar os recursos necessários do NLTK
Execute este script antes de rodar sua análise de sentimentos
"""

import nltk
import ssl

def baixar_recursos_nltk():
    """Baixa todos os recursos necessários do NLTK"""
    
    # Contorna problemas de SSL em alguns sistemas
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    print("Baixando recursos do NLTK...")
    print("=" * 50)
    
    # Lista de recursos essenciais para análise de sentimentos
    recursos = [
        'punkt',           # Tokenizador de sentenças
        'punkt_tab',       # Nova versão do punkt
        'stopwords',       # Palavras irrelevantes
        'vader_lexicon',   # Léxico para análise de sentimentos
        'wordnet',         # Base de dados lexical
        'omw-1.4',         # Multilingual wordnet
        'rslp',            # Stemmer para português
        'floresta',        # Corpus em português
        'mac_morpho',      # Corpus morfológico do português
        'averaged_perceptron_tagger',  # POS tagger
    ]
    
    # Baixa cada recurso
    for recurso in recursos:
        try:
            print(f"Baixando {recurso}...")
            nltk.download(recurso, quiet=False)
            print(f"✓ {recurso} baixado com sucesso!")
        except Exception as e:
            print(f"✗ Erro ao baixar {recurso}: {e}")
    
    print("\n" + "=" * 50)
    print("Download concluído!")
    
    # Testa se os recursos foram instalados corretamente
    print("\nTestando recursos...")
    try:
        from nltk.tokenize import word_tokenize, sent_tokenize
        from nltk.corpus import stopwords
        from nltk.sentiment import SentimentIntensityAnalyzer
        
        # Teste básico
        texto_teste = "Este é um teste. Vamos verificar se tudo funciona!"
        tokens = word_tokenize(texto_teste)
        sentencas = sent_tokenize(texto_teste)
        
        print("✓ Tokenização funcionando!")
        print(f"  Tokens: {tokens}")
        print(f"  Sentenças: {sentencas}")
        
        # Teste stopwords
        stop_words = stopwords.words('portuguese')
        print(f"✓ Stopwords carregadas! ({len(stop_words)} palavras)")
        
        # Teste VADER
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores("I love this!")
        print(f"✓ VADER funcionando! Scores: {scores}")
        
    except Exception as e:
        print(f"✗ Erro no teste: {e}")
        return False
    
    print("\n🎉 Todos os recursos do NLTK foram instalados e testados com sucesso!")
    return True

if __name__ == "__main__":
    baixar_recursos_nltk()