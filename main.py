from agent import AIAgent

def main():
    agent = AIAgent()
    print("AI Agent: I can help you search for news, process documents, or chat about them. I can help you send or read emails too. Try saying 'send email' or 'read emails'.")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("AI Agent: Goodbye!")
            break

        response = agent.chat(user_input)
        print(f"AI Agent: {response}")

if __name__ == "__main__":
    main()




################ Discord ###################
# import discord
# from agent import AIAgent

# # Initialize the Discord client
# intents = discord.Intents.default()
# intents.message_content = True  # Enable reading message content
# client = discord.Client(intents=intents)

# # Initialize your AI agent
# agent = AIAgent()

# @client.event
# async def on_ready():
#     print(f'Logged in as {client.user}')

# @client.event
# async def on_message(message):
#     # Ignore messages from the bot itself
#     if message.author == client.user:
#         return

#     # Process messages
#     user_input = message.content
#     if user_input.lower() in ["exit", "quit"]:
#         await message.channel.send("AI Agent: Goodbye!")
#     else:
#         # Get response from your AI agent
#         response = agent.chat(user_input)
#         await message.channel.send(f"AI Agent: {response}")

# # Replace 'YOUR_TOKEN_HERE' with your bot token
# client.run('YOUR_TOKEN_HERE')