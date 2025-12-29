#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import sys

from google import genai
from google.genai import types

def prompt_enhancer(text: str, api_key: str = None) -> str:
    """Enhance a natural language command into a precise image editing prompt.
    
    Args:
        text: The original instruction text
        api_key: Gemini API key (if not provided, will try to get from environment)
    
    Returns:
        Enhanced prompt text
    """
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not provided and not found in environment")
        sys.exit(1)
    
    client = genai.Client(api_key=api_key)
    
    agent_prompt = \
"""
You are an expert-level AI assistant specializing in robotics and computer vision. Your task is to convert a natural language command for a robot into a precise, task-focused prompt for an AI image editing model. The prompt must describe only the final state of the objects directly involved in the manipulation.

When converting a command, you must adhere strictly to the following rules:

1. Start with an Edit Command:
The prompt must always begin with: Edit the image.

2. Describe the Final State of Involved Objects:
Focus exclusively on the static, final scene. Describe the spatial relationship between the object that was moved (the "moved object") and the object it interacted with (the "static object"). Use precise, unambiguous language.

Example: Instead of "Move the cup next to the plate," describe the result: The goal image shows the cup positioned directly next to the plate on its right side.

3. Enforce Object Consistency:
State that the physical properties of the involved objects do not change. The prompt should include: The intrinsic properties (e.g., shape, color, size) of the [moved object] and the [static object] remain unchanged.

4. Re-emphasize the Core Interaction (Mandatory):
At the very end of the prompt, add a concluding sentence to explicitly restate the roles of the key objects. This reinforces the core task. The format should be: To be clear: the [moved object] has been moved, and the [static object] remains in its original position.

5. Strict Exclusion:

Absolutely no mention of robots, arms, or grippers.

Do not mention other uninvolved objects or the general environment (e.g., "other objects remain unchanged," "the lighting is the same"). The prompt must be completely focused on the primary objects of the task.

Strict Examples to Follow:

User's Robot Command:
Stack the green cube on top of the orange cube.

Your Generated Prompt (V2):
Edit the image. The goal image shows the green cube resting securely on the center of the top surface of the orange cube. The intrinsic properties (e.g., shape, color, size) of the green cube and the orange cube remain unchanged. To be clear: the green cube has been moved, and the orange cube remains in its original position.

User's Robot Command:
Put the apple in the box.

Your Generated Prompt (V2):
Edit the image. The goal image shows the apple located inside the box. The intrinsic properties (e.g., shape, color, size) of the apple and the box remain unchanged. To be clear: the apple has been moved, and the box remains in its original position.

User's Robot Command:
Place the screwdriver flush with the left edge of the wrench.

Your Generated Prompt (V2):
Edit the image. The goal image shows the screwdriver aligned flush with the left edge of the wrench. The intrinsic properties (e.g., shape, color, size) of the screwdriver and the wrench remain unchanged. To be clear: the screwdriver has been moved, and the wrench remains in its original position.

Original prompt: {original_prompt}

Please provide an enhanced version of this prompt that will generate better, more detailed images. Return only the enhanced prompt without additional explanation.
"""

    try:
        enhanced_prompt = agent_prompt.format(original_prompt=text)
        
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[types.Content(
                role="user",
                parts=[types.Part(text=enhanced_prompt)]
            )],
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=2048
            )
        )
        
        if hasattr(response, 'text') and response.text:
            enhanced_text = response.text.strip()
            print(f"Original prompt: {text}")
            print(f"Enhanced prompt: {enhanced_text}")
            return enhanced_text
        else:
            print("⚠️ No enhanced text returned, using original prompt")
            return text
            
    except Exception as e:
        print(f"❌ Prompt enhancement failed: {e}")
        print("Using original prompt")
        return text


def main():
    parser = argparse.ArgumentParser(description="Enhance natural language commands into precise image editing prompts")
    parser.add_argument("text", type=str, help="Input text instruction to enhance")
    parser.add_argument("--api-key", type=str, help="Gemini API key (optional, will use GEMINI_API_KEY env var if not provided)")
    args = parser.parse_args()
    
    enhanced = prompt_enhancer(args.text, args.api_key)
    print(f"\nEnhanced text: {enhanced}")


if __name__ == "__main__":
    main()
