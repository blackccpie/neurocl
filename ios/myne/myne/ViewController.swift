//
//  ViewController.swift
//  myne
//
//  Created by Albert Murienne on 12/03/2017.
//  Copyright © 2017 Blackccpie. All rights reserved.
//

import UIKit

class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    
    @IBOutlet weak var StartCamera: UIButton!
    
    @IBOutlet weak var ImageDisplay: UIImageView!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }

    
    @IBAction func StartCameraAction(sender: UIButton) {
        
        if UIImagePickerController.isSourceTypeAvailable(
            UIImagePickerControllerSourceType.Camera) {
                
                let picker = UIImagePickerController()
                picker.delegate = self
                picker.sourceType = .Camera
        
                presentViewController(picker, animated: true, completion: nil)
                
        }
    }
    
    func imagePickerController(picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : AnyObject]) {
        
        ImageDisplay.image = info[UIImagePickerControllerOriginalImage] as? UIImage;
        dismissViewControllerAnimated(true, completion: nil)
        
    }
}

