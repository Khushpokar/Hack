from django import forms

class ChatForm(forms.Form):
    user_input=forms.CharField(max_length=500,widget=forms.TextInput(attrs={'placeholder':'Type your message, I will help you.'}))
    