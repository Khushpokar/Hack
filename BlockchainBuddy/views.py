from django.shortcuts import render, HttpResponse
from .forms import ChatForm 
from chatbot.chatbot import process_user_input
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db.models import Q
from django.contrib.auth import authenticate, login, logout

chat_history = [] 


@csrf_exempt
def home(request):
    return HttpResponse("This is Home page")

# Create your views here.

@csrf_exempt
def chat_page(request):
    # form = ChatForm(request.POST)
    if request.method == 'POST':
        # user_input = form.cleaned_data['user_input']
        bot_response = process_user_input(request.POST.get('user_input'))

        # Save chat history statically
        chat_history.append({'user': request.POST.get('user_input'), 'bot': bot_response})
        return render(request, 'chat_page.html', { 'bot_response': bot_response, 'chat_history': chat_history})

        # return render(request, 'D:/BlockchainBuddy/BlockchainBuddy/templates/chat_page.html', {'form': form, 'bot_response': bot_response, 'chat_history': chat_history})

    return render(request, 'chat_page.html', { 'chat_history': chat_history})


