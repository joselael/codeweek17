from flask import Flask, render_template, url_for, redirect
from app import app

@app.route('/')
def index():
	#homepage
	return render_template('index.html')