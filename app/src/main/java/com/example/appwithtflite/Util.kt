package com.example.appwithtflite

import android.graphics.drawable.Drawable
import android.net.Uri
import android.widget.ImageView
import com.bumptech.glide.Glide
import com.bumptech.glide.request.RequestListener

fun ImageView.loadImage(imageUri: Uri){
    Glide.with(this.context)
        .load(imageUri)
        .into(this)
}

fun ImageView.loadImage(imageUri: Uri, listener:RequestListener<Drawable>){
    Glide.with(this.context)
        .load(imageUri)
        .listener(listener)
        .into(this)
}