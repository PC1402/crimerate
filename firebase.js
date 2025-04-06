// firebase.js
import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";

const firebaseConfig = {
  apiKey: "AIzaSyAEoYyr1UngbT1UkPl9ZfNhpmCP3tD62cQ",
  authDomain: "crimerateprediction-d5098.firebaseapp.com",
  projectId: "crimerateprediction-d5098",
  storageBucket: "crimerateprediction-d5098.firebasestorage.app",
  messagingSenderId: "365884272529",
  appId: "1:365884272529:web:8a386b1c1addcbf3e23430",
  measurementId: "G-L5VK0Y8WEP"
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

export { auth };

