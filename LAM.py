"""
WhatsApp Web Large Action Model (LAM) using Microsoft's Phi-2
This script creates a LAM for controlling WhatsApp Web through a browser automation
framework on macOS.
"""

import os
import time
import json
import argparse
from typing import Dict, List, Any, Tuple, Optional, Union
import logging

# Core dependencies
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, ElementNotInteractableException
from webdriver_manager.chrome import ChromeDriverManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WhatsAppLAM:
    """Large Action Model for controlling WhatsApp Web."""
    
    def __init__(self, model_name: str = "microsoft/phi-2"):
        """
        Initialize the WhatsApp LAM.
        
        Args:
            model_name: The name or path of the language model to use
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)
        
        # Initialize browser
        self.driver = None
        self.wait = None
        
        # Define available actions
        self.actions = {
            "open_whatsapp": self.open_whatsapp,
            "send_message": self.send_message,
            "search_contact": self.search_contact,
            "get_unread_messages": self.get_unread_messages,
            "close_whatsapp": self.close_whatsapp
        }
        
        logger.info("WhatsApp LAM initialized successfully")
    
    def initialize_browser(self):
        """Initialize and set up the browser for WhatsApp Web."""
        if self.driver is not None:
            return
            
        logger.info("Initializing Chrome browser")
        options = webdriver.ChromeOptions()
        options.add_argument("--start-maximized")
        options.add_argument("--disable-notifications")
        # Uncomment below to run headless (no UI)
        # options.add_argument("--headless")
        
        # Initialize the driver
        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options
        )
        self.wait = WebDriverWait(self.driver, 30)
        
        logger.info("Browser initialized")
    
    def open_whatsapp(self) -> Dict[str, Any]:
        """
        Open WhatsApp Web in the browser.
        
        Returns:
            Dictionary with status message
        """
        self.initialize_browser()
        
        logger.info("Opening WhatsApp Web")
        self.driver.get("https://web.whatsapp.com/")
        
        try:
            # Wait for the QR code to be scanned or for WhatsApp to load
            self.wait.until(EC.presence_of_element_located(
                (By.XPATH, '//div[@data-testid="chat-list"]')
            ))
            logger.info("WhatsApp Web opened successfully")
            return {"status": "success", "message": "WhatsApp Web opened and loaded"}
        except TimeoutException:
            logger.warning("Timeout waiting for WhatsApp Web to load")
            return {
                "status": "pending", 
                "message": "Please scan the QR code on the browser to log in to WhatsApp Web"
            }
    
    def search_contact(self, query: str) -> Dict[str, Any]:
        """
        Search for a contact on WhatsApp.
        
        Args:
            query: Name or number to search for
            
        Returns:
            Dictionary with search results
        """
        if not self.driver:
            return {"status": "error", "message": "Browser not initialized"}
        
        logger.info(f"Searching for contact: {query}")
        
        try:
            # Click on the search box
            search_box = self.wait.until(EC.presence_of_element_located(
                (By.XPATH, '//div[@data-testid="chat-list-search"]')
            ))
            search_box.click()
            
            # Enter search query
            input_box = self.wait.until(EC.presence_of_element_located(
                (By.XPATH, '//div[@data-testid="chat-list-search-input"]//div[@contenteditable="true"]')
            ))
            input_box.clear()
            input_box.send_keys(query)
            
            # Wait for search results
            time.sleep(2)
            
            # Get search results
            chat_items = self.driver.find_elements(By.XPATH, '//div[@data-testid="cell-frame-container"]')
            
            results = []
            for item in chat_items[:5]:  # Limit to first 5 results
                try:
                    name = item.find_element(By.XPATH, './/span[@data-testid="cell-frame-title"]').text
                    results.append({"name": name})
                except:
                    continue
            
            logger.info(f"Found {len(results)} contacts")
            return {"status": "success", "results": results}
            
        except Exception as e:
            logger.error(f"Error searching for contact: {str(e)}")
            return {"status": "error", "message": f"Error searching for contact: {str(e)}"}
    
    def send_message(self, contact_name: str, message: str) -> Dict[str, Any]:
        """
        Send a message to a specific contact.
        
        Args:
            contact_name: Name of the contact to send message to
            message: Content of the message to send
            
        Returns:
            Dictionary with status message
        """
        if not self.driver:
            return {"status": "error", "message": "Browser not initialized"}
        
        logger.info(f"Sending message to: {contact_name}")
        
        try:
            # First search for the contact
            search_result = self.search_contact(contact_name)
            if search_result["status"] != "success" or len(search_result["results"]) == 0:
                return {"status": "error", "message": f"Contact '{contact_name}' not found"}
            
            # Click on the first search result
            chat_items = self.driver.find_elements(By.XPATH, '//div[@data-testid="cell-frame-container"]')
            if len(chat_items) > 0:
                chat_items[0].click()
                time.sleep(1)
            
            # Find and click the message input box
            message_input = self.wait.until(EC.presence_of_element_located(
                (By.XPATH, '//div[@data-testid="conversation-compose-box-input"]//div[@contenteditable="true"]')
            ))
            message_input.click()
            message_input.send_keys(message)
            
            # Send the message
            send_button = self.wait.until(EC.presence_of_element_located(
                (By.XPATH, '//button[@data-testid="compose-btn-send"]')
            ))
            send_button.click()
            
            logger.info("Message sent successfully")
            return {"status": "success", "message": f"Message sent to {contact_name}"}
            
        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")
            return {"status": "error", "message": f"Error sending message: {str(e)}"}
    
    def get_unread_messages(self) -> Dict[str, Any]:
        """
        Get all unread messages.
        
        Returns:
            Dictionary with unread messages
        """
        if not self.driver:
            return {"status": "error", "message": "Browser not initialized"}
        
        logger.info("Looking for unread messages")
        
        try:
            # Find chats with unread messages (they have a notification badge)
            unread_chats = self.driver.find_elements(By.XPATH, '//span[@data-testid="icon-unread-count"]/..')
            
            unread_messages = []
            
            for chat in unread_chats:
                try:
                    # Get the parent chat item
                    chat_item = chat.find_element(By.XPATH, './ancestor::div[@data-testid="cell-frame-container"]')
                    
                    # Get contact name
                    contact_name = chat_item.find_element(By.XPATH, './/span[@data-testid="cell-frame-title"]').text
                    
                    # Get unread count
                    unread_count = chat.find_element(By.XPATH, './/span[@data-testid="icon-unread-count"]').text
                    
                    # Click on the chat to view messages
                    chat_item.click()
                    time.sleep(1)
                    
                    # Get the most recent messages
                    message_elements = self.driver.find_elements(
                        By.XPATH, '//div[contains(@data-testid, "msg-container")]'
                    )
                    
                    recent_messages = []
                    # Get the last 5 messages or less
                    for msg_elem in message_elements[-min(5, len(message_elements)):]:
                        try:
                            author = "You"
                            if "message-in" in msg_elem.get_attribute("class"):
                                author = contact_name
                            
                            message_text = msg_elem.find_element(By.XPATH, './/div[@data-testid="msg-text"]').text
                            recent_messages.append({
                                "author": author,
                                "text": message_text
                            })
                        except:
                            continue
                    
                    unread_messages.append({
                        "contact": contact_name,
                        "unread_count": unread_count,
                        "recent_messages": recent_messages
                    })
                    
                    # Go back to the main chat list
                    back_button = self.driver.find_element(By.XPATH, '//span[@data-testid="back"]')
                    back_button.click()
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error processing unread chat: {str(e)}")
                    continue
            
            logger.info(f"Found {len(unread_messages)} chats with unread messages")
            return {"status": "success", "unread_messages": unread_messages}
            
        except Exception as e:
            logger.error(f"Error getting unread messages: {str(e)}")
            return {"status": "error", "message": f"Error getting unread messages: {str(e)}"}
    
    def close_whatsapp(self) -> Dict[str, Any]:
        """
        Close the WhatsApp Web browser session.
        
        Returns:
            Dictionary with status message
        """
        if self.driver:
            logger.info("Closing WhatsApp Web browser")
            self.driver.quit()
            self.driver = None
            self.wait = None
            return {"status": "success", "message": "WhatsApp Web browser closed"}
        return {"status": "info", "message": "No active browser session to close"}
    
    def generate_response(self, prompt: str) -> str:
        """
        Generate a response using the Phi-2 model.
        
        Args:
            prompt: Input prompt for the model
            
        Returns:
            The model's response
        """
        # Prepare the prompt
        full_prompt = f"""You are a helpful WhatsApp automation assistant. 
Based on the user's request, determine what action to take.
Available actions are:
- open_whatsapp
- search_contact (requires 'query' parameter)
- send_message (requires 'contact_name' and 'message' parameters)
- get_unread_messages
- close_whatsapp

User request: {prompt}

Respond in JSON format with 'action' and 'parameters' fields.
"""
        
        # Tokenize and generate
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=512,
                temperature=0.2,
                top_p=0.9,
                do_sample=True
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract JSON from the response
        response = response.replace(full_prompt, "")
        try:
            # Find JSON in the response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json_str
            else:
                logger.warning("Could not find JSON in model response")
                return '{"action": null, "parameters": {}, "error": "Invalid model response format"}'
        except Exception as e:
            logger.error(f"Error parsing model response: {str(e)}")
            return '{"action": null, "parameters": {}, "error": "Error parsing model response"}'
    
    def execute_action(self, action_data: str) -> Dict[str, Any]:
        """
        Execute an action based on the parsed response.
        
        Args:
            action_data: JSON string containing action and parameters
            
        Returns:
            Result of the executed action
        """
        try:
            data = json.loads(action_data)
            action = data.get("action")
            parameters = data.get("parameters", {})
            
            if action not in self.actions:
                return {"status": "error", "message": f"Unknown action: {action}"}
            
            logger.info(f"Executing action: {action}")
            return self.actions[action](**parameters)
            
        except json.JSONDecodeError:
            logger.error("Invalid JSON in action data")
            return {"status": "error", "message": "Invalid JSON in action data"}
        except Exception as e:
            logger.error(f"Error executing action: {str(e)}")
            return {"status": "error", "message": f"Error executing action: {str(e)}"}
    
    def process_request(self, user_request: str) -> Dict[str, Any]:
        """
        Process a user request by generating a response and executing the action.
        
        Args:
            user_request: Natural language request from the user
            
        Returns:
            Result of the executed action
        """
        logger.info(f"Processing user request: {user_request}")
        
        # Generate response
        response = self.generate_response(user_request)
        logger.info(f"Generated response: {response}")
        
        # Execute action
        result = self.execute_action(response)
        
        return result

def main():
    """Main entry point for the WhatsApp LAM CLI."""
    parser = argparse.ArgumentParser(description="WhatsApp Web Large Action Model using Phi-2")
    parser.add_argument("--model", type=str, default="microsoft/phi-2", help="Model name or path")
    args = parser.parse_args()
    
    # Initialize the LAM
    lam = WhatsAppLAM(model_name=args.model)
    
    print("WhatsApp Web LAM initialized. Enter commands or requests (type 'exit' to quit):")
    while True:
        user_input = input("> ")
        if user_input.lower() in ["exit", "quit"]:
            if lam.driver:
                lam.close_whatsapp()
            break
        
        # Process the request
        result = lam.process_request(user_input)
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()