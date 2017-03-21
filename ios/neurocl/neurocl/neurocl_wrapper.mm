//
//  neurocl_wrapper.m
//  neurocl
//
//  Created by Albert Murienne on 14/03/2017.
//  Copyright © 2017 Blackccpie. All rights reserved.
//

#import <Foundation/Foundation.h>

#import "neurocl_wrapper.h"
#import "neurocl.h"

#include "imagetools/ocr.h"

@interface NeuroclWrapper ()

@property (nonatomic, readonly) std::shared_ptr<neurocl::network_manager_interface> net_manager;

@end

@implementation NeuroclWrapper

- (instancetype) init
{
    return [self initWithNet:@"topology-mnist-kaggle.txt" weights:@"weights-mnist-kaggle.bin"];
}

- (instancetype) initWithNet:(NSString*) topology weights:(NSString*) weights
{
    self = [super init];

    if(self != nil)
    {
        logger_manager& lm = logger_manager::instance();
        lm.add_logger( policy_type::cout, "neurocl_frmk" );

        std::string str_topology([topology UTF8String]);
        std::string str_weights([weights UTF8String]);

        _net_manager = neurocl::network_factory::build();
        _net_manager->load_network( str_topology, str_weights );
    }

    return self;
}

- (void) dealloc
{
    _net_manager.reset();
}

+ (UIImage *) convertUIImage32ToGray8:(UIImage *) image_in {
    
    CGImageRef imageRef = image_in.CGImage;
    
    size_t bitsPerPixel = CGImageGetBitsPerPixel(imageRef);
    size_t bitsPerComponent = CGImageGetBitsPerComponent(imageRef);
    size_t width = CGImageGetWidth(imageRef);
    size_t height = CGImageGetHeight(imageRef);
    
    CGImageAlphaInfo a = CGImageGetAlphaInfo(imageRef);
    
    NSAssert(bitsPerPixel == 32 && bitsPerComponent == 8 && a == kCGImageAlphaNoneSkipLast, @"unsupported image type supplied");
    
    CGContextRef targetImage = CGBitmapContextCreate(NULL, width, height, 8, 1 * width, CGColorSpaceCreateDeviceGray(), kCGImageAlphaNone);
    
    UInt32 *sourceData = (UInt32*)[((__bridge_transfer NSData*) CGDataProviderCopyData(CGImageGetDataProvider(imageRef))) bytes];
    UInt8 *targetData = (UInt8 *)CGBitmapContextGetData(targetImage);
    
    size_t offset = 0;
    
    std::for_each( targetData, targetData+(width*height),
        [&sourceData,&offset](UInt8& val)
        {
            UInt8 *sourceDataPtr = reinterpret_cast<UInt8 *>( &sourceData[offset] );
            val = static_cast<float>( ( sourceDataPtr[0] + sourceDataPtr[1] + sourceDataPtr[2] ) / 3 );
            ++offset;
        }
    );
    
    CGImageRef newImageRef = CGBitmapContextCreateImage(targetImage);
    UIImage *newImage = [UIImage imageWithCGImage:newImageRef scale:[image_in scale] orientation: image_in.imageOrientation];
    
    CGContextRelease(targetImage);
    CGImageRelease(newImageRef);
    
    return newImage;
}

+ (void) convertUIImage8ToArray:(UIImage *) image_in image_out:(float *) image_out {

    CGImageRef imageRef = image_in.CGImage;

    size_t bitsPerPixel = CGImageGetBitsPerPixel(imageRef);
    size_t bitsPerComponent = CGImageGetBitsPerComponent(imageRef);
    size_t width = CGImageGetWidth(imageRef);
    size_t height = CGImageGetHeight(imageRef);

    CGImageAlphaInfo a = CGImageGetAlphaInfo(imageRef);

    NSAssert(bitsPerPixel == 8 && bitsPerComponent == 8 && a == kCGImageAlphaNone, @"unsupported image type supplied");

    UInt8 *sourceData = (UInt8*)[((__bridge_transfer NSData*) CGDataProviderCopyData(CGImageGetDataProvider(imageRef))) bytes];

    size_t offset = 0;

    std::for_each( image_out, image_out+(width*height),
        [&sourceData,&offset](float& val)
        {
            val = static_cast<float>( sourceData[offset] );
            ++offset;
        }
    );
}

- (NSString*) digit_recognizer:(UIImage*) in
{
    int wi = in.size.width * in.scale;
    int hi = in.size.height * in.scale;

    boost::shared_array<float> input( new float[wi*hi] );

    NSLog(@"digit reco input image is %ix%i", wi, hi );

    [[self class] convertUIImage8ToArray:in image_out:input.get()];

    ocr_helper helper( _net_manager );
    helper.process( input.get(), wi, hi );

    return [NSString stringWithCString:helper.reco_string().c_str()
            encoding:[NSString defaultCStringEncoding]];
}

@end
