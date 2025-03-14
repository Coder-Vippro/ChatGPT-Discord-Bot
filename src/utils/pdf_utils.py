import io
import asyncio
import discord
import logging
from typing import List, Dict, Any, Optional, Tuple
from PyPDF2 import PdfReader
from src.config.config import PDF_BATCH_SIZE
from src.utils.openai_utils import trim_content_to_token_limit

async def process_pdf(message: discord.Message, pdf_content: bytes, user_prompt: str, model: str, client) -> None:
    """
    Process a PDF file with improved error handling and token management.
    
    Args:
        message: Discord message object for responses
        pdf_content: Binary PDF content
        user_prompt: User query about the PDF
        model: OpenAI model to use
        client: OpenAI client
    """
    try:
        pdf_file = io.BytesIO(pdf_content)
        pdf_reader = PdfReader(pdf_file)
        pages_content = []
        
        # Extract text from PDF
        for page_num, page in enumerate(pdf_reader.pages, 1):
            text = page.extract_text()
            if text.strip():  # Only add non-empty pages
                pages_content.append({
                    "page": page_num,
                    "content": text.strip()
                })
                
        if not pages_content:
            await message.channel.send("Error: Could not extract any text from the PDF.")
            return

        # Initial batch size
        total_pages = len(pages_content)
        current_batch_size = PDF_BATCH_SIZE
        processed_pages = 0

        # Handle single-page PDF
        if total_pages == 1:
            batch_content = f"\nPDF Page 1:\n{pages_content[0]['content']}\n"
            await process_pdf_batch(
                model=model,
                client=client,
                user_prompt=user_prompt,
                batch_content=batch_content,
                current_batch=1,
                total_batches=1,
                channel=message.channel
            )
            return
        
        while current_batch_size > 0 and processed_pages < total_pages:
            try:
                remaining_pages = total_pages - processed_pages
                total_batches = (remaining_pages + current_batch_size - 1) // current_batch_size
                await message.channel.send(f"Processing PDF with {remaining_pages} remaining pages in {total_batches} batches...")
                
                batch_start = processed_pages
                success = True
                
                for i in range(batch_start, total_pages, current_batch_size):
                    batch = pages_content[i:i+current_batch_size]
                    batch_content = ""
                    for page_data in batch:
                        page_num = page_data["page"]
                        content = page_data["content"]
                        batch_content += f"\nPDF Page {page_num}:\n{content}\n"
                    
                    current_batch = (i - batch_start) // current_batch_size + 1
                    success = await process_pdf_batch(
                        model=model,
                        client=client,
                        user_prompt=user_prompt,
                        batch_content=batch_content,
                        current_batch=current_batch,
                        total_batches=total_batches,
                        channel=message.channel
                    )
                    
                    if not success:
                        # If batch processing failed, reduce batch size and retry from current position
                        current_batch_size = current_batch_size // 2
                        if current_batch_size > 0:
                            await message.channel.send(f"Reducing batch size to {current_batch_size} pages and retrying...")
                        else:
                            await message.channel.send("Cannot process PDF. Batch size reduced to minimum.")
                            return
                    else:
                        processed_pages += len(batch)
                        await asyncio.sleep(2)  # Delay between successful batches
                
                if success and processed_pages >= total_pages:
                    await message.channel.send("PDF processing completed successfully!")
                    return
                    
            except Exception as e:
                current_batch_size = current_batch_size // 2
                if current_batch_size > 0:
                    await message.channel.send(f"Error occurred. Reducing batch size to {current_batch_size} pages and retrying...")
                else:
                    await message.channel.send(f"Error processing PDF: {str(e)}")
                    return
                    
    except Exception as e:
        await message.channel.send(f"Error processing PDF: {str(e)}")
        return

async def process_pdf_batch(model: str, client, user_prompt: str, batch_content: str, 
                            current_batch: int, total_batches: int, channel, max_retries=3) -> bool:
    """
    Process a single batch of PDF content with auto-adjustment for token limits.
    
    Args:
        model: OpenAI model to use
        client: OpenAI client
        user_prompt: User query about the PDF
        batch_content: Content of the current batch
        current_batch: Current batch number
        total_batches: Total number of batches
        channel: Discord channel for responses
        max_retries: Maximum number of retries
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    from src.config.config import PDF_ANALYSIS_PROMPT
    
    batch_size = len(batch_content.split('\n'))
    original_content = batch_content
    
    for attempt in range(max_retries):
        try:
            # Create message without history but with appropriate prompt handling
            trimmed_content = trim_content_to_token_limit(batch_content, 7000)  # Leave room for prompt
            
            messages = []
            if model in ["o1-mini", "o1-preview"]:
                # These models don't support system prompts
                messages = [
                    {"role": "user", "content": f"Instructions: {PDF_ANALYSIS_PROMPT}\n\n{user_prompt}\n\nAnalyze the following content:\n{trimmed_content}"}
                ]
            else:
                messages = [
                    {"role": "system", "content": PDF_ANALYSIS_PROMPT},
                    {"role": "user", "content": f"{user_prompt}\n\nAnalyze the following content:\n{trimmed_content}"}
                ]
            
            # Add await here - this was the issue causing the error
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1
            )
            
            reply = response.choices[0].message.content
            batch_response = f"Batch {current_batch}/{total_batches} (Pages in batch: {batch_size}):\n{reply}"
            await send_response(channel, batch_response)
            return True
            
        except Exception as e:
            error_str = str(e)
            if "413" in error_str and attempt < max_retries - 1:
                # Split the batch content in half and try again
                content_parts = batch_content.split('\n')
                mid = len(content_parts) // 2
                batch_content = '\n'.join(content_parts[:mid])
                batch_size = len(batch_content.split('\n'))
                await channel.send(f"Batch {current_batch} was too large, reducing size and retrying...")
                continue
            elif attempt == max_retries - 1:
                await channel.send(f"Error processing batch {current_batch}: {str(e)}")
                return False
    return False

async def send_response(channel: discord.TextChannel, reply: str):
    """
    Send a response to the Discord channel, handling long responses.
    
    Args:
        channel: Discord channel to send the response to
        reply: Text to send
    """
    # Safety check - ensure reply is not empty
    if not reply or not reply.strip():
        reply = "I'm sorry, I couldn't generate a proper response. Please try again."
    
    if len(reply) > 2000:
        with open("response.txt", "w", encoding="utf-8") as file:
            file.write(reply)
        await channel.send(
            "The response was too long, so it has been saved to a file.",
            file=discord.File("response.txt")
        )
    else:
        await channel.send(reply)