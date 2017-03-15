//
//  ViewController.swift
//  myne
//
//  Created by Albert Murienne on 12/03/2017.
//  Copyright © 2017 Blackccpie. All rights reserved.
//

import UIKit

func getEnvironmentVar(_ name: String) -> String? {
    guard let rawValue = getenv(name) else { return nil }
    return String(utf8String: rawValue)
}

class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {

    @IBOutlet weak var StartCamera: UIButton!

    @IBOutlet weak var ImageDisplay: UIImageView!

    @IBOutlet weak var RecognizedText: UITextField!

    @IBOutlet weak var RecognizerActivity: UIActivityIndicatorView!

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }

    @IBAction func StartCameraAction(_ sender: UIButton) {

        if UIImagePickerController.isSourceTypeAvailable(
            UIImagePickerControllerSourceType.camera) {

                let picker = UIImagePickerController()
                picker.delegate = self
                picker.sourceType = .camera

                present(picker, animated: true, completion: nil)

        }
    }

    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : Any]) {

        //self.ImageDisplay.image = info[UIImagePickerControllerOriginalImage] as? UIImage;
        self.ImageDisplay.image = NeuroclWrapper.convertUIImage32(toGray8: info[UIImagePickerControllerOriginalImage] as? UIImage );
        
        self.dismiss(animated: true, completion: nil)
        
        self.RecognizerActivity.startAnimating()
        
        DispatchQueue.global(qos: .background).async {
            
            let neuroclResPath: String! = getEnvironmentVar( "NEUROCL_RESOURCE_PATH" )
            let neuroclWrapper: NeuroclWrapper = NeuroclWrapper(net: neuroclResPath+"/topology-mnist-kaggle.txt", weights: neuroclResPath+"/weights-mnist-kaggle.bin")

            let label: String = neuroclWrapper.digit_recognizer( self.ImageDisplay.image )
            
            //let label: String = "1234567890"
            //sleep(4)
            
            self.RecognizedText.text = label
            
            DispatchQueue.main.async {
                
                self.RecognizerActivity.stopAnimating()
            }
        }
    }
}

