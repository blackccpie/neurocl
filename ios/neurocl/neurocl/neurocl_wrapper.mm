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

+ (void) convertUIImageToGray8:(UIImage *) image_in image_out:(float *) image_out {

    CGImageRef imageRef = image_in.CGImage;

    size_t bitsPerPixel = CGImageGetBitsPerPixel(imageRef);
    size_t bitsPerComponent = CGImageGetBitsPerComponent(imageRef);
    size_t width = CGImageGetWidth(imageRef);
    size_t height = CGImageGetHeight(imageRef);

    CGImageAlphaInfo a = CGImageGetAlphaInfo(imageRef);

    NSAssert(bitsPerPixel == 32 && bitsPerComponent == 8 && a == kCGImageAlphaNoneSkipLast, @"unsupported image type supplied");

    UInt32 *sourceData = (UInt32*)[((__bridge_transfer NSData*) CGDataProviderCopyData(CGImageGetDataProvider(imageRef))) bytes];
    UInt32 *sourceDataPtr;

    UInt8 r,g,b;
    size_t offset;
    for (uint y = 0; y < height; y++)
    {
        for (uint x = 0; x < width; x++)
        {
            offset = y * width + x;

            if (offset+2 < width * height)
            {
                sourceDataPtr = &sourceData[y * width + x];

                r = sourceDataPtr[0+0];
                g = sourceDataPtr[0+1];
                b = sourceDataPtr[0+2];

                image_out[y * width + x] = static_cast<float>( (r+g+b) / 3 );
            }
        }
    }
}

- (NSString*) digit_recognizer:(UIImage*) in
{
    int wi = in.size.width * in.scale;
    int hi = in.size.height * in.scale;

    boost::shared_array<float> input( new float[wi*hi] );

    NSLog(@"digit reco input image is %ix%i", wi, hi );

    [[self class] convertUIImageToGray8:in image_out:input.get()];

    ocr_helper helper( _net_manager );
    helper.process( input.get(), wi, hi );

    return [NSString stringWithCString:helper.reco_string().c_str()
            encoding:[NSString defaultCStringEncoding]];
}

@end
