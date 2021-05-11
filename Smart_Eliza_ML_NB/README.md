# Eliza chatbot in Python with ML model 

This is based on original Eliza code from the github https://github.com/nlpia/eliza

This version of Eliza applied the Machine Learning technologies.

The ML model will make the sentiment analysis for the user's each single input sentence
and if the mood is negative, the chatbox will return "I am sorry to hear that ,how I can help you?"
otherwise the chat will return as original rule based message to user.

## Usage

Can be run interactively:

```
$ python eliza_with_ml.py
How do you do.  Please tell me your problem.
> I would like to have a chat bot.
You say you would like to have a chat bot ?
> bye
Goodbye.  Thank you for talking to me.
```

[comment]: <> (...or imported and used as a library:)

[comment]: <> (```python)

[comment]: <> (import eliza)

[comment]: <> (eliza = eliza.Eliza&#40;&#41;)

[comment]: <> (eliza.load&#40;'doctor.txt'&#41;)

[comment]: <> (print&#40;eliza.initial&#40;&#41;&#41;)

[comment]: <> (while True:)

[comment]: <> (    said = input&#40;'> '&#41;)

[comment]: <> (    response = eliza.respond&#40;said&#41;)

[comment]: <> (    if response is None:)

[comment]: <> (        break)

[comment]: <> (    print&#40;response&#41;)

[comment]: <> (print&#40;eliza.final&#40;&#41;&#41;)

[comment]: <> (```)