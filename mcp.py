import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
import time
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import wikipedia
import chromadb
import os
import json
from bs4 import BeautifulSoup

class StockAnalyzerMCP:
    """
    Enhanced Model Context Protocol implementation for stock analysis with Llama 3.2 1B.
    Features:
    - Stock data retrieval and analysis via Yahoo Finance
    - Web access including Wikipedia integration
    - ChromaDB for persistent memory and document storage
    """
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_chunk_size: int = 512,
        overlap_size: int = 64,
        attention_sink_size: int = 4,
        chroma_db_path: str = "./chroma_db",
        embedding_layer_name: str = "base_model.model.embed_tokens",
    ):
        """
        Initialize the enhanced Stock Analyzer MCP
        
        Args:
            model_name: HuggingFace model ID
            device: Device to run on (cuda/cpu)
            max_chunk_size: Maximum chunk size for processing
            overlap_size: Size of overlap between chunks
            attention_sink_size: Number of tokens to keep in attention sink
            chroma_db_path: Path to store ChromaDB files
            embedding_layer_name: Name of embedding layer for custom extraction
        """
        print(f"Loading model {model_name} on {device}...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        
        # Configuration
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.attention_sink_size = attention_sink_size
        self.embedding_layer_name = embedding_layer_name
        
        # Track KV cache state
        self.kv_cache = None
        self.cached_token_count = 0
        
        # Find embedding layer for efficient token retrieval
        self.embedding_layer = self._get_embedding_layer()
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        
        # Create collections for different types of data
        self.stock_collection = self.chroma_client.get_or_create_collection("stock_data")
        self.news_collection = self.chroma_client.get_or_create_collection("news_data")
        self.wiki_collection = self.chroma_client.get_or_create_collection("wiki_data")
        
        print("Stock Analyzer MCP initialized successfully!")
        
    def _get_embedding_layer(self):
        """Retrieve the embedding layer from the model"""
        parts = self.embedding_layer_name.split('.')
        module = self.model
        for part in parts:
            module = getattr(module, part)
        return module
    
    #----------------------------------------
    # Core MCP Implementation
    #----------------------------------------
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text to token IDs"""
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
    
    def decode_tokens(self, tokens: torch.Tensor) -> str:
        """Decode token IDs to text"""
        return self.tokenizer.decode(tokens[0], skip_special_tokens=True)
    
    def get_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Get embeddings for token IDs"""
        return self.embedding_layer(token_ids)
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks for processing"""
        # Tokenize text
        tokens = self.tokenizer.encode(text)
        
        if len(tokens) <= self.max_chunk_size:
            return [text]
        
        # Create overlapping chunks
        chunks = []
        for i in range(0, len(tokens), self.max_chunk_size - self.overlap_size):
            chunk_tokens = tokens[i:i + self.max_chunk_size]
            chunks.append(self.tokenizer.decode(chunk_tokens))
            
        return chunks
    
    def process_chunk(
        self, 
        chunk: str, 
        is_first_chunk: bool = False,
        return_hidden_states: bool = False
    ) -> Dict[str, Any]:
        """Process a single chunk of text"""
        # Reset KV cache if this is the first chunk
        if is_first_chunk:
            self.kv_cache = None
            self.cached_token_count = 0
        
        # Encode chunk
        input_ids = self.encode_text(chunk)
        
        # Forward pass with KV cache
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                use_cache=True,
                past_key_values=self.kv_cache,
                output_hidden_states=return_hidden_states
            )
        
        # Update KV cache
        self.kv_cache = outputs.past_key_values
        self.cached_token_count += input_ids.size(1)
        
        # Get hidden states for embeddings if requested
        if return_hidden_states:
            # Use last layer hidden states for embeddings
            embeddings = outputs.hidden_states[-1].mean(dim=1)
        else:
            embeddings = None
        
        return {
            "logits": outputs.logits,
            "hidden_states": outputs.hidden_states if return_hidden_states else None,
            "embeddings": embeddings
        }
    
    def generate_with_mcp(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        chunk_processing: bool = True
    ) -> str:
        """Generate text using MCP"""
        # Process prompt in chunks if needed
        if chunk_processing and len(self.tokenizer.encode(prompt)) > self.max_chunk_size:
            chunks = self.chunk_text(prompt)
            
            # Process all chunks except the last one
            for i, chunk in enumerate(chunks[:-1]):
                self.process_chunk(chunk, is_first_chunk=(i == 0))
                
            # Process the last chunk to get final state
            input_ids = self.encode_text(chunks[-1])
        else:
            # Process the entire prompt at once
            input_ids = self.encode_text(prompt)
            self.kv_cache = None
            self.cached_token_count = 0
        
        # Track generated tokens
        all_token_ids = input_ids.clone()
        
        # Generation loop
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids if self.kv_cache is None else input_ids[:, -1:],
                    use_cache=True,
                    past_key_values=self.kv_cache
                )
            
            # Get logits for the next token
            next_token_logits = outputs.logits[:, -1, :]
            
            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample from distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Update input ids, only keep the last token for efficiency
            input_ids = next_token
            all_token_ids = torch.cat([all_token_ids, next_token], dim=1)
            
            # Update KV cache
            self.kv_cache = outputs.past_key_values
            self.cached_token_count += 1
            
            # Stop if EOS token is generated
            if next_token[0, 0].item() == self.tokenizer.eos_token_id:
                break
            
            # Apply attention sink if needed
            if self.cached_token_count > self.model.config.max_position_embeddings - max_new_tokens:
                self._apply_attention_sink()
                
        # Decode the entire generation
        return self.decode_tokens(all_token_ids)
    
    def _apply_attention_sink(self):
        """Apply attention sink mechanism to maintain a bounded context"""
        # Keep the first attention_sink_size tokens and the most recent ones
        preserve_size = self.max_chunk_size - self.attention_sink_size
        
        # Update KV cache to keep only the important tokens
        for layer_idx in range(len(self.kv_cache)):
            # Each layer has a tuple of (keys, values)
            k, v = self.kv_cache[layer_idx]
            
            # For each head in the layer
            for head_idx in range(len(k)):
                # Keep attention sink tokens and most recent tokens
                k[head_idx] = torch.cat([
                    k[head_idx][:, :self.attention_sink_size],
                    k[head_idx][:, -preserve_size:]
                ], dim=1)
                
                v[head_idx] = torch.cat([
                    v[head_idx][:, :self.attention_sink_size],
                    v[head_idx][:, -preserve_size:]
                ], dim=1)
        
        # Update token count in the cache
        self.cached_token_count = self.attention_sink_size + preserve_size
    
    #----------------------------------------
    # Stock Data and Analysis Functions
    #----------------------------------------
    
    def get_stock_data(
        self, 
        ticker: str, 
        period: str = "1y", 
        interval: str = "1d",
        store_in_memory: bool = True
    ) -> pd.DataFrame:
        """
        Fetch stock data from Yahoo Finance
        
        Args:
            ticker: Stock ticker symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            store_in_memory: Whether to store data in ChromaDB
            
        Returns:
            DataFrame with stock data
        """
        try:
            # Fetch data using yfinance
            stock = yf.Ticker(ticker)
            data = stock.history(period=period, interval=interval)
            
            # Save data to memory if requested
            if store_in_memory and not data.empty:
                # Get metadata
                info = stock.info
                company_name = info.get('shortName', ticker)
                
                # Convert dataframe to string for storage
                data_str = data.to_csv()
                
                # Create a document ID
                doc_id = f"{ticker}_{period}_{interval}_{datetime.now().strftime('%Y%m%d')}"
                
                # Store in ChromaDB
                self.stock_collection.upsert(
                    documents=[data_str],
                    metadatas=[{
                        "ticker": ticker,
                        "company_name": company_name,
                        "period": period,
                        "interval": interval,
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "data_type": "stock_price"
                    }],
                    ids=[doc_id]
                )
                
                print(f"Stored stock data for {ticker} in memory with ID: {doc_id}")
            
            return data
            
        except Exception as e:
            print(f"Error fetching stock data for {ticker}: {e}")
            return pd.DataFrame()
    
    def analyze_stock_technical(self, ticker: str, period: str = "1y") -> Dict[str, Any]:
        """
        Perform technical analysis on a stock
        
        Args:
            ticker: Stock ticker symbol
            period: Time period to analyze
            
        Returns:
            Dictionary with technical indicators
        """
        # Get stock data
        data = self.get_stock_data(ticker, period)
        
        if data.empty:
            return {"error": f"No data available for {ticker}"}
        
        # Calculate simple technical indicators
        results = {}
        
        # Moving averages
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['MA200'] = data['Close'].rolling(window=200).mean()
        
        # MACD
        data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = data['EMA12'] - data['EMA26']
        data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        # RSI (Relative Strength Index)
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Get current values
        current = data.iloc[-1]
        prev = data.iloc[-2] if len(data) > 1 else current
        
        # Calculate metrics
        results['current_price'] = current['Close']
        results['price_change'] = (current['Close'] - prev['Close']) / prev['Close'] * 100
        results['ma20'] = current['MA20']
        results['ma50'] = current['MA50'] 
        results['ma200'] = current['MA200']
        results['rsi'] = current['RSI']
        results['macd'] = current['MACD']
        results['macd_signal'] = current['Signal']
        
        # Simple trend analysis
        results['trend_20day'] = "bullish" if current['Close'] > current['MA20'] else "bearish"
        results['trend_50day'] = "bullish" if current['Close'] > current['MA50'] else "bearish"
        results['trend_200day'] = "bullish" if current['Close'] > current['MA200'] else "bearish"
        
        # Interpret RSI
        if current['RSI'] > 70:
            results['rsi_status'] = "overbought"
        elif current['RSI'] < 30:
            results['rsi_status'] = "oversold" 
        else:
            results['rsi_status'] = "neutral"
            
        # MACD interpretation
        if current['MACD'] > current['Signal']:
            results['macd_status'] = "bullish"
        else:
            results['macd_status'] = "bearish"
            
        # Store analysis in memory
        analysis_str = json.dumps(results)
        doc_id = f"{ticker}_analysis_{datetime.now().strftime('%Y%m%d')}"
        
        self.stock_collection.upsert(
            documents=[analysis_str],
            metadatas=[{
                "ticker": ticker,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "data_type": "technical_analysis"
            }],
            ids=[doc_id]
        )
        
        return results
    
    def get_company_info(self, ticker: str) -> Dict[str, Any]:
        """
        Get company information for a stock ticker
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with company information
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Extract relevant information
            relevant_info = {
                "name": info.get("shortName", "N/A"),
                "long_name": info.get("longName", "N/A"),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "website": info.get("website", "N/A"),
                "description": info.get("longBusinessSummary", "N/A"),
                "market_cap": info.get("marketCap", "N/A"),
                "pe_ratio": info.get("trailingPE", "N/A"),
                "dividend_yield": info.get("dividendYield", "N/A") * 100 if info.get("dividendYield") else "N/A",
                "52wk_high": info.get("fiftyTwoWeekHigh", "N/A"),
                "52wk_low": info.get("fiftyTwoWeekLow", "N/A"),
                "avg_volume": info.get("averageVolume", "N/A"),
                "beta": info.get("beta", "N/A"),
                "currency": info.get("currency", "USD")
            }
            
            # Store in ChromaDB
            doc_id = f"{ticker}_info_{datetime.now().strftime('%Y%m%d')}"
            info_str = json.dumps(relevant_info)
            
            self.stock_collection.upsert(
                documents=[info_str],
                metadatas=[{
                    "ticker": ticker,
                    "company_name": relevant_info["name"],
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "data_type": "company_info"
                }],
                ids=[doc_id]
            )
            
            return relevant_info
            
        except Exception as e:
            print(f"Error fetching company info for {ticker}: {e}")
            return {"error": str(e)}
    
    #----------------------------------------
    # Wikipedia and Web Access Functions
    #----------------------------------------
    
    def get_wikipedia_info(self, topic: str, store_in_memory: bool = True) -> str:
        """
        Get information from Wikipedia
        
        Args:
            topic: Topic to search on Wikipedia
            store_in_memory: Whether to store in ChromaDB
            
        Returns:
            Wikipedia content as string
        """
        try:
            # Check if we already have this in memory
            results = self.wiki_collection.query(
                query_texts=[topic],
                n_results=1
            )
            
            if results and results['documents'] and results['documents'][0]:
                print(f"Retrieved Wikipedia info for '{topic}' from memory")
                return results['documents'][0][0]
            
            # Search for the page
            search_results = wikipedia.search(topic)
            if not search_results:
                return f"No Wikipedia information found for '{topic}'"
            
            # Get the most relevant page
            try:
                page = wikipedia.page(search_results[0], auto_suggest=False)
            except wikipedia.DisambiguationError as e:
                # If disambiguation, take the first option
                page = wikipedia.page(e.options[0], auto_suggest=False)
                
            # Get content
            content = page.content
            
            # Store in memory if requested
            if store_in_memory:
                doc_id = f"{topic.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d')}"
                
                self.wiki_collection.upsert(
                    documents=[content],
                    metadatas=[{
                        "topic": topic,
                        "title": page.title,
                        "url": page.url,
                        "date_accessed": datetime.now().strftime("%Y-%m-%d"),
                        "data_type": "wikipedia"
                    }],
                    ids=[doc_id]
                )
                
                print(f"Stored Wikipedia data for '{topic}' in memory with ID: {doc_id}")
            
            return content
            
        except Exception as e:
            print(f"Error fetching Wikipedia info for '{topic}': {e}")
            return f"Error retrieving information: {str(e)}"
    
    def get_web_content(self, url: str, store_in_memory: bool = True) -> str:
        """
        Fetch content from a web URL
        
        Args:
            url: Web URL to fetch
            store_in_memory: Whether to store in ChromaDB
            
        Returns:
            Web content as string
        """
        try:
            # Fetch the web page
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
                
            # Get text
            text = soup.get_text(separator='\n')
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            # Store in memory if requested
            if store_in_memory:
                # Create a simple URL-based ID
                url_id = url.replace("https://", "").replace("http://", "").replace("/", "_").replace(".", "_")
                doc_id = f"{url_id[:50]}_{datetime.now().strftime('%Y%m%d')}"
                
                # Get title if available
                title = soup.title.string if soup.title else url
                
                self.news_collection.upsert(
                    documents=[text],
                    metadatas=[{
                        "url": url,
                        "title": title,
                        "date_accessed": datetime.now().strftime("%Y-%m-%d"),
                        "data_type": "web_content"
                    }],
                    ids=[doc_id]
                )
                
                print(f"Stored web content from {url} in memory with ID: {doc_id}")
            
            return text
            
        except Exception as e:
            print(f"Error fetching web content from {url}: {e}")
            return f"Error retrieving content: {str(e)}"
    
    def get_stock_news(self, ticker: str, limit: int = 5, store_in_memory: bool = True) -> List[Dict]:
        """
        Get news articles related to a stock
        
        Args:
            ticker: Stock ticker symbol
            limit: Maximum number of news articles
            store_in_memory: Whether to store in ChromaDB
            
        Returns:
            List of news article dictionaries
        """
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            
            # Limit the number of articles
            news = news[:limit] if limit else news
            
            # Process each news item
            processed_news = []
            for item in news:
                article = {
                    "title": item.get("title", ""),
                    "publisher": item.get("publisher", ""),
                    "link": item.get("link", ""),
                    "publish_time": datetime.fromtimestamp(item.get("providerPublishTime", 0)),
                    "type": item.get("type", ""),
                    "summary": item.get("summary", "")
                }
                processed_news.append(article)
                
                # Store in memory if requested
                if store_in_memory:
                    doc_id = f"{ticker}_news_{item.get('uuid', datetime.now().strftime('%Y%m%d%H%M%S'))}"
                    
                    self.news_collection.upsert(
                        documents=[article["summary"]],
                        metadatas=[{
                            "ticker": ticker,
                            "title": article["title"],
                            "publisher": article["publisher"],
                            "url": article["link"],
                            "date": article["publish_time"].strftime("%Y-%m-%d"),
                            "data_type": "stock_news"
                        }],
                        ids=[doc_id]
                    )
            
            return processed_news
            
        except Exception as e:
            print(f"Error fetching news for {ticker}: {e}")
            return []
    
    #----------------------------------------
    # Memory Management Functions
    #----------------------------------------
    
    def create_text_embedding(self, text: str) -> np.ndarray:
        """
        Create embedding for text using the model
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array with embedding
        """
        # Process text through model to get hidden states
        chunks = self.chunk_text(text)
        embeddings = []
        
        for i, chunk in enumerate(chunks):
            output = self.process_chunk(chunk, is_first_chunk=(i==0), return_hidden_states=True)
            if output["embeddings"] is not None:
                embeddings.append(output["embeddings"].cpu().numpy())
        
        if embeddings:
            # Average embeddings from all chunks
            return np.mean(embeddings, axis=0).squeeze()
        else:
            return np.array([])
    
    def search_memory(
        self, 
        query: str, 
        collection: str = "all", 
        n_results: int = 5
    ) -> Dict[str, Any]:
        """
        Search memory for relevant information
        
        Args:
            query: Search query
            collection: Collection to search ("stock_data", "news_data", "wiki_data", or "all")
            n_results: Number of results to return
            
        Returns:
            Dictionary with search results
        """
        results = {}
        
        collections_to_search = []
        if collection == "all" or collection == "stock_data":
            collections_to_search.append(("stock_data", self.stock_collection))
        if collection == "all" or collection == "news_data":
            collections_to_search.append(("news_data", self.news_collection))
        if collection == "all" or collection == "wiki_data":
            collections_to_search.append(("wiki_data", self.wiki_collection))
            
        for coll_name, coll in collections_to_search:
            try:
                coll_results = coll.query(
                    query_texts=[query],
                    n_results=n_results
                )
                
                results[coll_name] = {
                    "documents": coll_results["documents"][0] if coll_results["documents"] else [],
                    "metadatas": coll_results["metadatas"][0] if coll_results["metadatas"] else [],
                    "ids": coll_results["ids"][0] if coll_results["ids"] else [],
                    "distances": coll_results["distances"][0] if coll_results["distances"] else []
                }
            except Exception as e:
                print(f"Error searching collection {coll_name}: {e}")
                results[coll_name] = {"error": str(e)}
                
        return results
    
    def remember_analysis(
        self, 
        content: str, 
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Store analysis result in memory
        
        Args:
            content: Analysis content to store
            metadata: Metadata for the content
            
        Returns:
            Document ID
        """
        if metadata is None:
            metadata = {}
            
        # Add timestamp if not present
        if "date" not in metadata:
            metadata["date"] = datetime.now().strftime("%Y-%m-%d")
        if "data_type" not in metadata:
            metadata["data_type"] = "analysis"
            
        # Create ID
        base_id = metadata.get("ticker", "analysis") 
        doc_id = f"{base_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Store in appropriate collection
        if "ticker" in metadata:
            self.stock_collection.upsert(
                documents=[content],
                metadatas=[metadata],
                ids=[doc_id]
            )
        else:
            # Default to news collection for general analysis
            self.news_collection.upsert(
                documents=[content],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
        return doc_id
    
    #----------------------------------------
    # High-Level Analysis Functions
    #----------------------------------------
    
    def analyze_stock(self, ticker: str) -> Dict[str, Any]:
        """
        Comprehensive stock analysis
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with analysis results
        """
        results = {}
        
        # Get basic data
        print(f"Analyzing stock: {ticker}")
        results["company_info"] = self.get_company_info(ticker)
        
        # Get technical analysis
        print("Performing technical analysis...")
        results["technical"] = self.analyze_stock_technical(ticker)
        
        # Get news
        print("Fetching recent news...")
        results["news"] = self.get_stock_news(ticker)
        
        # Get wiki info if available
        try:
            company_name = results["company_info"]["long_name"]
            print(f"Fetching Wikipedia information for {company_name}...")
            wiki_info = self.get_wikipedia_info(company_name)
            # Only keep the first 1000 characters for summary
            results["wiki_summary"] = wiki_info[:1000] + "..." if len(wiki_info) > 1000 else wiki_info
        except Exception as e:
            print(f"Error fetching Wikipedia info: {e}")
            results["wiki_summary"] = f"No Wikipedia information available: {str(e)}"
        
        # Generate a narrative summary using the model
        print("Generating analysis summary...")
        analysis_context = f"""
        Stock Analysis for {ticker} ({results['company_info'].get('name', ticker)})
        
        Company Info:
        - Sector: {results['company_info'].get('sector', 'N/A')}
        - Industry: {results['company_info'].get('industry', 'N/A')}
        - Market Cap: {results['company_info'].get('market_cap', 'N/A')}
        - P/E Ratio: {results['company_info'].get('pe_ratio', 'N/A')}
        
        Technical Analysis:
        - Current Price: {results['technical'].get('current_price', 'N/A')}
        - 20-day trend: {results['technical'].get('trend_20day', 'N/A')}
        - 50-day trend: {results['technical'].get('trend_50day', 'N/A')}
        - RSI: {results['technical'].get('rsi', 'N/A')} ({results['technical'].get('rsi_status', 'N/A')})
        - MACD status: {results['technical'].get('macd_status', 'N/A')}
        
        Recent News Headlines:
        """
        
        # Add news headlines
        for i, news_item in enumerate(results['news'][:3], 1):
            analysis_context += f"\n{i}. {news_item['title']}"
            
        analysis_context += "\n\nBased on the above information, provide a brief analysis of this stock's current situation and outlook."
        
        # Use MCP to generate a summary
        summary = self.generate_with_mcp(
            analysis_context,
            max_new_tokens=300,
            temperature=0.7
        )
        
        # Store the generated summary
        results["summary"] = summary
        self.remember_analysis(
            summary,
            metadata={
                "ticker": ticker,
                "company_name": results['company_info'].get('name', ticker),
                "data_type": "stock_analysis_summary"
            }
        )
        
        return results
    
    def compare_stocks(self, tickers: List[str]) -> str:
        """
        Compare multiple stocks and generate insights
        
        Args:
            tickers: List of stock ticker symbols
            
        Returns:
            Comparison analysis as string
        """
        print(f"Comparing stocks: {', '.join(tickers)}")
        
        # Analyze each stock
        analyses = {}
        for ticker in tickers:
            analyses[ticker] = self.analyze_stock(ticker)
            
        # Create comparison context
        comparison_context = f"Stock Comparison Analysis for {', '.join(tickers)}\n\n"
        
        # Add comparison table
        comparison_context += "| Metric | " + " | ".join(tickers) + " |\n"
        comparison_context += "|" + "----|" * (len(tickers) + 1) + "\n"
        
        # Add key metrics
        metrics = [
            ("Current Price", lambda t: analyses[t]["technical"].get("current_price", "N/A")),
            ("Market Cap", lambda t: analyses[t]["company_info"].get("market_cap", "N/A")),
            ("P/E Ratio", lambda t: analyses[t]["company_info"].get("pe_ratio", "N/A")),
            ("Dividend Yield", lambda t: analyses[t]["company_info"].get("dividend_yield", "N/A")),
            ("52wk High", lambda t: analyses[t]["company_info"].get("52wk_high", "N/A")),
            ("52wk Low", lambda t: analyses[t]["company_info"].get("52wk_low", "N/A")),
            ("RSI", lambda t: analyses[t]["technical"].get("rsi", "N/A")),
            ("20-day trend", lambda t: analyses[t]["technical"].get("trend_20day", "N/A")),
            ("50-day trend", lambda t: analyses[t]["technical"].get("trend_50day", "N/A"))
        ]
        
        for metric_name, metric_func in metrics:
            comparison_context += f"| {metric_name} | " + " | ".join(str(metric_func(t)) for t in tickers) + " |\n"
            
        comparison_context += "\nBased on the above comparison, analyze these stocks and identify which might be the better investment opportunity. Consider their relative performance, financial health, and market position."
        
        # Generate comparison analysis
        comparison_analysis = self.generate_with_mcp(
            comparison_context,
            max_new_tokens=500,
            temperature=0.7
        )
        
        # Remember the comparison
        self.remember_analysis(
            comparison_analysis,
            metadata={
                "tickers": ",".join(tickers),
                "data_type": "stock_comparison"
            }
        )
        
        return comparison_analysis
    
    def market_summary(self) -> str:
        """
        Generate an overall market summary
        
        Returns:
            Market summary text
        """
        print("Generating market summary...")
        
        # Get major indices data
        indices = ["^GSPC", "^DJI", "^IXIC", "^RUT"]  # S&P 500, Dow Jones, NASDAQ, Russell 2000
        index_data = {}
        
        for index in indices:
            data = self.get_stock_data(index, period="5d")
            if not data.empty:
                current = data.iloc[-1]
                prev = data.iloc[-2] if len(data) > 1 else current
                change_pct = (current['Close'] - prev['Close']) / prev['Close'] * 100
                
                index_data[index] = {
                    "name": {
                        "^GSPC": "S&P 500",
                        "^DJI": "Dow Jones",
                        "^IXIC": "NASDAQ",
                        "^RUT": "Russell 2000"
                    }.get(index, index),
                    "current": current['Close'],
                    "change_pct": change_pct,
                    "direction": "up" if change_pct > 0 else "down",
                    "volume": current['Volume']
                }
        
        # Get some trending news
        market_news = []
        try:
            # Most active stocks as a proxy for market news
            market_movers = yf.Tickers("^GSPC").tickers["^GSPC"].news
            market_news = [item.get("title", "") for item in market_movers[:5]]
        except Exception as e:
            print(f"Error fetching market news: {e}")
            market_news = ["Unable to fetch market news"]
        
        # Create context for the model
        context = "Current Market Summary\n\n"
        
        # Add index data
        context += "Major Indices:\n"
        for idx, data in index_data.items():
            direction = "▲" if data["direction"] == "up" else "▼"
            context += f"- {data['name']}: {data['current']:.2f} {direction} ({data['change_pct']:.2f}%)\n"
            
        # Add news
        context += "\nRecent Market News:\n"
        for i, headline in enumerate(market_news, 1):
            context += f"{i}. {headline}\n"
            
        context += "\nBased on the above information, provide a brief summary of the current market conditions, possible factors affecting the market, and a general outlook."
        
        # Generate summary
        summary = self.generate_with_mcp(
            context,
            max_new_tokens=400,
            temperature=0.7
        )
        
        # Remember the summary
        self.remember_analysis(
            summary,
            metadata={
                "data_type": "market_summary",
                "date": datetime.now().strftime("%Y-%m-%d")
            }
        )
        
        return summary
    
    def query_stock_knowledge(self, query: str) -> str:
        """
        Answer a question about stocks using all available knowledge
        
        Args:
            query: User's query about stocks
            
        Returns:
            Response to the query
        """
        print(f"Processing query: {query}")
        
        # Search memory for relevant information
        memory_results = self.search_memory(query, collection="all", n_results=5)
        
        # Build context for the query
        context = f"Query: {query}\n\nRelevant Information:\n\n"
        
        # Add memory results if available
        for collection_name, results in memory_results.items():
            if "error" in results:
                continue
                
            documents = results.get("documents", [])
            metadatas = results.get("metadatas", [])
            
            for i, (doc, meta) in enumerate(zip(documents, metadatas)):
                if not doc or not meta:
                    continue
                    
                # Add metadata info
                context += f"--- {collection_name.upper()} DOCUMENT {i+1} ---\n"
                context += f"Type: {meta.get('data_type', 'Unknown')}\n"
                
                if 'ticker' in meta:
                    context += f"Ticker: {meta.get('ticker')}\n"
                if 'company_name' in meta:
                    context += f"Company: {meta.get('company_name')}\n"
                if 'title' in meta:
                    context += f"Title: {meta.get('title')}\n"
                if 'date' in meta:
                    context += f"Date: {meta.get('date')}\n"
                    
                context += "\nContent:\n"
                
                # For long documents, add just the beginning
                if len(doc) > 500:
                    context += doc[:500] + "...\n"
                else:
                    context += doc + "\n"
                    
                context += "\n"
                
        # Add instruction to answer the query
        context += f"\nBased on the above information, please answer the query: {query}\n"
        context += "If the information provided is not sufficient to answer the query, please indicate what additional information would be needed."
        
        # Generate response
        response = self.generate_with_mcp(
            context,
            max_new_tokens=500,
            temperature=0.7
        )
        
        return response


# Example usage
if __name__ == "__main__":
    # Initialize the Stock Analyzer
    analyzer = StockAnalyzerMCP(
        model_name="distilbert-base-uncased",
        max_chunk_size=512,
        overlap_size=64,
        attention_sink_size=4,
        chroma_db_path="./stock_analysis_db"
    )
    
    print("\n--- STOCK ANALYZER MCP ---\n")
    
    # Example 1: Analyze a single stock
    print("\n1. Analyzing a single stock (AAPL)...")
    analysis = analyzer.analyze_stock("AAPL")
    print("\nAnalysis Summary:")
    print(analysis["summary"])
    print("-" * 80)
    
    # Example 2: Compare two stocks
    print("\n2. Comparing stocks (MSFT vs GOOG)...")
    comparison = analyzer.compare_stocks(["MSFT", "GOOG"])
    print("\nComparison Analysis:")
    print(comparison)
    print("-" * 80)
    
    # Example 3: Get market summary
    print("\n3. Generating market summary...")
    market = analyzer.market_summary()
    print("\nMarket Summary:")
    print(market)
    print("-" * 80)
    
    # Example 4: Ask a specific question
    print("\n4. Answering a specific question...")
    question = "Which tech stock has better growth potential based on recent earnings?"
    answer = analyzer.query_stock_knowledge(question)
    print(f"\nQ: {question}")
    print(f"A: {answer}")
    print("-" * 80)