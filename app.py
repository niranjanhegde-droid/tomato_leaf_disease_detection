from flask import Flask, render_template, request, redirect, session, url_for, flash
import csv
import os
import classify
from PIL import Image
import numpy as np
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import IntegrityError
import bcrypt
from datetime import datetime
import pytz  # Import for timezone support

app = Flask(__name__)
port = int(os.getenv('PORT', 5000))
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///Model.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = '1A2bc4s'
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(128), unique=True, nullable=False)
    email = db.Column(db.String(128), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)

    def __repr__(self) -> str:
        return f'<User {self.username}>'

    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email
        }

class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    userId = db.Column(db.Integer, db.ForeignKey('user.id'))
    image = db.Column(db.String(128), nullable=False)
    result = db.Column(db.String(128), nullable=False)
    model_used = db.Column(db.String(50), nullable=False)  # ✅ NEW FIELD
    date_created = db.Column(db.DateTime, default=datetime.utcnow)



    def __repr__(self) -> str:
        return f'<Result {self.result} - {self.userId}>'

    def to_dict(self):
        ist = pytz.timezone('Asia/Kolkata')
        local_time = self.date_created.replace(tzinfo=pytz.utc).astimezone(ist)
        formatted_time = local_time.strftime('%d-%m-%Y %I:%M %p')
        return {
            'id': self.id,
            'userId': self.userId,
            'image': self.image,
            'result': self.result,
            'model_used': self.model_used,  
            'date_created': formatted_time
        }


with app.app_context():
    db.create_all()

def getUser():
    userId = session.get('userId')
    data = User.query.get(userId)
    if data is None:
        return None, []
    details = data.to_dict()
    history_data = History.query.filter(History.userId == userId).order_by(History.date_created.desc()).all()
    history = [item.to_dict() for item in history_data]
    return details, history


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/model')
def model():
    return render_template("model.html")

@app.route('/dataset')
def dataset():
    return render_template("dataset.html")

@app.route('/remedies')
def remedies():
    return render_template("remedies.html")

@app.route('/logout')
def logout():
    session['logged'] = False
    session.pop('userId', None)
    return redirect('/')

@app.route('/update', methods=['POST'])
def update():
    userId = session['userId']
    username = request.form.get('name')
    email = request.form.get('email')
    user = User.query.get(userId)
    user.username = username
    user.email = email
    db.session.commit()
    return redirect('/profile')

@app.route('/changePassword', methods=['POST'])
def changePassword():
    userId = session['userId']
    oldPass = request.form.get('oldPass')
    newPass = request.form.get('newPass')
    confPass = request.form.get('confPass')
    user = User.query.filter_by(id=userId).first()
    details, history = getUser()
    if user:
        password_match = bcrypt.checkpw(oldPass.encode('utf-8'), user.password)
        if password_match:
            if newPass == confPass:
                salt = bcrypt.gensalt()
                password_hash = bcrypt.hashpw(newPass.encode('utf-8'), salt)
                user.password = password_hash
                db.session.commit()
                return redirect('/profile')
            else:
                return render_template('profile.html', details=details, history_items=history, newPassError="New Passwords do not match")
        else:
            return render_template('profile.html', details=details, history_items=history, oldPassError="Old Password is Incorrect")
    return redirect('/login')

@app.route('/profile')
def profile():
    details, history = getUser()
    if details is None:
        return redirect('/login')
    return render_template('profile.html', details=details, history_items=history)

@app.route('/output', methods=['POST', 'GET'])
def output():
    image_path = ''
    result = None
    modelNo = 0  # Default to CNN

    if request.method == 'POST':
        image_file = request.files["imagefile"]
        modelNo = int(request.form.get('model_no'))

        image_file.save(f'static/images/test/{image_file.filename}')
        image_path = "static/images/test/" + image_file.filename

        image = Image.open(image_file)
        predicted_class, confidence = classify.predict(image, modelNo)
        result = f"{predicted_class} with a {confidence:.2f}% Confidence."

        userId = session['userId']
        model_used = "CNN" if modelNo == 0 else "EfficientNet"
        new_history = History(userId=userId, result=result, image=image_path, model_used=model_used)
        db.session.add(new_history)
        db.session.commit()

    return render_template("output.html", result=result, image=image_path, selected_model=modelNo)


@app.route('/register', methods=['POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('name')
        email = request.form.get('email')
        newPass = request.form.get('newPass')
        confPass = request.form.get('confPass')

        if not username or not confPass or not newPass or not email:
            return render_template('login.html', error="Please enter all required fields")

        if newPass == confPass:
            try:
                salt = bcrypt.gensalt()
                password_hash = bcrypt.hashpw(confPass.encode('utf-8'), salt)
                new_user = User(username=username, email=email, password=password_hash)
                db.session.add(new_user)
                db.session.commit()
                return redirect('/login')
            except IntegrityError:
                return render_template('login.html', error="username or email already exists")
        else:
            return render_template('login.html', error="Passwords do not match")

@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    elif request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if not username or not password:
            return render_template('login.html', error="Send all the required data")

        user = User.query.filter_by(username=username).first()
        if not user:
            return render_template('login.html', error="No User Found")

        password_match = bcrypt.checkpw(password.encode('utf-8'), user.password)
        if password_match:
            session['logged'] = True
            flash("Login Successful ✅", "success")
            session['userId'] = user.id
            return redirect('/output')
        else:
            return render_template('login.html', success="Login successful!")

@app.route('/delete_history/<int:history_id>', methods=['POST'])
def delete_history(history_id):
    history_item = History.query.get(history_id)
    if history_item and history_item.userId == session.get('userId'):
        if os.path.exists(history_item.image):
            os.remove(history_item.image)
        db.session.delete(history_item)
        db.session.commit()
    return redirect('/profile')

@app.route('/delete_all_history', methods=['POST'])
def delete_all_history():
    user_id = session.get('userId')
    if user_id:
        all_history = History.query.filter_by(userId=user_id).all()
        for entry in all_history:
            if os.path.exists(entry.image):
                os.remove(entry.image)  # ✅ Delete image file from disk
            db.session.delete(entry)
        db.session.commit()
    return redirect('/profile')

if __name__ == '__main__':
    app.run(debug=True, port=port)
